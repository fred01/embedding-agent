#!/usr/bin/env python3
"""
Embedding agent for the book search system (fred01/search-lib).

Pulls work straight from the indexer, no queue in between:

    POST /api/agent/embeddings/lease    -> books with chunk texts, about TARGET_SECONDS of work for this agent
    (compute BGE-M3 dense vectors)
    POST /api/agent/embeddings/results  -> vectors; the indexer stores them in Qdrant and marks chunks READY
    POST /api/agent/embeddings/release  -> on shutdown, give unfinished books back right away

The next lease is fetched and the previous results are sent while the model is busy, so the GPU does not
wait for the network. If the agent dies, its leases expire on the indexer and the books are handed out again.

Configuration (environment):
    INDEXER_URL      indexer base URL (default https://book-indexer.svc.fred.org.ru)
    AGENT_TOKEN      bearer token (FACADE_TOKEN on the indexer); RS_HTTP_FACADE_TOKEN / FACADE_TOKEN also read
    WORKER_NAME      unique name of this agent (default: <hostname>-<device>)
    DEVICE           auto | mlx | cuda | cuda:N | mps | cpu   (auto: mlx on Apple Silicon; --cpu = DEVICE=cpu)
    FP16             auto | true | false                 (auto: fp16 on GPU/MPS, fp32 on CPU)
    BATCH_SIZE       model batch size (default: cuda 32, mlx/mps 16, cpu 4); lowered automatically on OOM
    TARGET_SECONDS   how much work to lease at once, in seconds of own throughput (default 120)
    DUTY_CYCLE       share of time the GPU computes, e.g. 0.3 (default 1): pauses between batches, quieter fans
    DUTY_CYCLE_FILE  file whose number overrides DUTY_CYCLE without a restart (re-read every 10 s)
    EMBED_URL        use an OpenAI-compatible /embeddings server (TEI, infinity, LiteLLM) instead of local torch
    EMBED_MODEL      model name for EMBED_URL (default BAAI/bge-m3), EMBED_API_KEY its key
    SKIP_REFERENCE_CHECK=true  skip the startup comparison with the FlagEmbedding reference vector
"""

import argparse
import os
import queue
import signal
import socket
import sys
import threading
import time
from typing import Dict, List, Optional

import requests

from embeddings import (
    EMBEDDING_DIMENSION,
    MODEL_NAME,
    HttpEmbedder,
    MlxEmbedder,
    TorchEmbedder,
    check_reference,
    detect_device,
    encode_vector,
    measure_rate,
    mlx_available,
)

VERSION = "2.1.0"
DEFAULT_INDEXER_URL = "https://book-indexer.svc.fred.org.ru"
SUBMIT_PART = 128          # chunks per POST /results (~700 KB of JSON)
MAX_CHUNKS_PER_LEASE = 2000


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class IndexerClient:
    def __init__(self, url: str, token: str, worker: str):
        self.url = url.rstrip("/") + "/api/agent/embeddings"
        self.worker = worker
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Accept-Encoding": "gzip"})

    def _post(self, path: str, body: dict, timeout: int, stop: threading.Event, attempts: int = 0) -> dict:
        """POST with retries on network errors and 5xx (attempts=0: until stop is set)."""
        delay = 2
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.session.post(f"{self.url}/{path}", json=body, timeout=timeout)
                if response.status_code < 500:
                    if response.status_code == 401:
                        raise SystemExit("Indexer rejected the token (401): check AGENT_TOKEN")
                    if response.status_code >= 400:
                        raise ValueError(f"{path}: HTTP {response.status_code} {response.text[:300]}")
                    return response.json()
                wait = int(response.headers.get("Retry-After", delay))
                log(f"{path}: HTTP {response.status_code}, retry in {wait}s")
            except (requests.ConnectionError, requests.Timeout) as e:
                wait = delay
                log(f"{path}: {type(e).__name__}, retry in {wait}s")
            if attempts and attempt >= attempts:
                raise ConnectionError(f"{path}: gave up after {attempt} attempts")
            if stop.wait(wait):
                raise ConnectionError(f"{path}: stopped")
            delay = min(delay * 2, 60)

    def info(self, stop: threading.Event) -> dict:
        delay = 2
        while True:
            try:
                response = self.session.get(f"{self.url}/info", timeout=30)
                if response.status_code == 401:
                    raise SystemExit("Indexer rejected the token (401): check AGENT_TOKEN")
                response.raise_for_status()
                return response.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                log(f"Indexer not reachable ({e}), retry in {delay}s")
                if stop.wait(delay):
                    raise SystemExit(0)
                delay = min(delay * 2, 60)

    def lease(self, max_chunks: int, rate: float, device: str, backend: str, stop: threading.Event) -> dict:
        return self._post("lease", {
            "worker": self.worker,
            "maxChunks": max_chunks,
            "chunksPerSec": round(rate, 3),
            "device": device,
            "backend": backend,
            "version": VERSION,
        }, timeout=120, stop=stop)

    def submit(self, results: List[dict], done: List[str], rate: float, stop: threading.Event) -> dict:
        return self._post("results", {
            "worker": self.worker,
            "model": MODEL_NAME,
            "results": results,
            "done": done,
            "chunksPerSec": round(rate, 3),
        }, timeout=600, stop=stop)

    def release(self) -> None:
        try:
            response = self.session.post(f"{self.url}/release", json={"worker": self.worker}, timeout=30)
            if response.ok:
                log(f"Released {response.json().get('booksReleased', 0)} leased books")
        except requests.RequestException as e:
            log(f"Release failed ({e}); leases will expire on their own")


class Agent:
    def __init__(self, client: IndexerClient, embedder, target_seconds: float, rate: float):
        self.client = client
        self.embedder = embedder
        self.target_seconds = target_seconds
        self.rate = rate  # chunks per second, exponential moving average of compute only
        self.stop = threading.Event()
        # set only when shutdown gives up on sending computed results
        self.abort = threading.Event()
        self.leases: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self.results: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2)
        self.total_chunks = 0
        self.started = time.time()

    def max_chunks(self) -> int:
        return int(min(MAX_CHUNKS_PER_LEASE, max(1, self.rate * self.target_seconds)))

    # -- threads -----------------------------------------------------------

    def fetch_loop(self) -> None:
        """Keeps one lease ready while the model works on the current one."""
        while not self.stop.is_set():
            try:
                lease = self.client.lease(self.max_chunks(), self.rate, self.embedder.device_label,
                                          self.embedder.backend, self.stop)
            except ConnectionError:
                continue
            except ValueError as e:
                log(f"Lease failed: {e}")
                self.stop.wait(30)
                continue
            if not lease.get("books"):
                wait = lease.get("retryAfterSeconds") or 60
                log(f"No work, asking again in {wait}s")
                self.stop.wait(wait)
                continue
            while not self.stop.is_set():
                try:
                    self.leases.put(lease, timeout=1)
                    break
                except queue.Full:
                    continue

    def submit_loop(self) -> None:
        while True:
            item = self.results.get()
            if item is None:
                return
            lease, vectors = item
            try:
                self.submit(lease, vectors)
            except ConnectionError as e:
                log(f"Results not sent ({e}); the books will be handed out again when the lease expires")
            except ValueError as e:
                log(f"Results rejected: {e}")

    def submit(self, lease: dict, vectors) -> None:
        """Send results in parts; a book is reported done in the part with its last chunk."""
        part: List[dict] = []
        done: List[str] = []
        i = 0
        for book in lease["books"]:
            for chunk in book["chunks"]:
                part.append({"bookId": book["bookId"], "chunkIndex": chunk["chunkIndex"],
                             "vector": encode_vector(vectors[i])})
                i += 1
                if len(part) >= SUBMIT_PART:
                    self.client.submit(part, done, self.rate, self.abort)
                    part, done = [], []
            done.append(book["bookId"])
        if part or done:
            self.client.submit(part, done, self.rate, self.abort)

    # -- main --------------------------------------------------------------

    def run(self) -> None:
        fetcher = threading.Thread(target=self.fetch_loop, name="lease", daemon=True)
        submitter = threading.Thread(target=self.submit_loop, name="submit", daemon=True)
        fetcher.start()
        submitter.start()
        try:
            while not self.stop.is_set():
                try:
                    lease = self.leases.get(timeout=1)
                except queue.Empty:
                    continue
                texts = [chunk["text"] for book in lease["books"] for chunk in book["chunks"]]
                started = time.time()
                vectors = self.embedder.embed(texts)
                elapsed = max(time.time() - started, 1e-6)
                assert vectors.shape == (len(texts), EMBEDDING_DIMENSION)

                self.rate = 0.7 * self.rate + 0.3 * (len(texts) / elapsed)
                self.total_chunks += len(texts)
                log(f"{len(lease['books'])} books, {len(texts)} chunks in {elapsed:.1f}s "
                    f"({len(texts) / elapsed:.2f}/s, avg {self.rate:.2f}/s) | total {self.total_chunks}")
                self.results.put((lease, vectors))
        finally:
            self.stop.set()
            self.results.put(None)
            # computed vectors are worth a couple of minutes of retries
            submitter.join(timeout=180)
            self.abort.set()
            submitter.join(timeout=10)
            self.client.release()


def build_embedder(args):
    embed_url = os.getenv("EMBED_URL")
    if embed_url:
        return HttpEmbedder(embed_url, model=os.getenv("EMBED_MODEL", MODEL_NAME),
                            api_key=os.getenv("EMBED_API_KEY"),
                            batch_size=int(os.getenv("BATCH_SIZE", "32")))

    device = os.getenv("DEVICE", "auto").strip().lower()
    if args.cpu or os.getenv("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        device = "cpu"
    if device in ("", "auto"):
        device = "mlx" if mlx_available() else detect_device()
    batch_size = int(os.getenv("BATCH_SIZE", "0")) or None
    if device == "mlx":
        return MlxEmbedder(batch_size=batch_size)
    fp16_env = os.getenv("FP16", "auto").lower()
    fp16 = None if fp16_env == "auto" else fp16_env in ("1", "true", "yes")
    return TorchEmbedder(device=device, fp16=fp16, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="BGE-M3 embedding agent for the book search indexer")
    parser.add_argument("--cpu", action="store_true", help="use CPU even if a GPU / Apple GPU is available")
    parser.add_argument("--benchmark", type=int, metavar="N", help="only measure speed on N chunks and exit")
    args = parser.parse_args()

    token = os.getenv("AGENT_TOKEN") or os.getenv("RS_HTTP_FACADE_TOKEN") or os.getenv("FACADE_TOKEN")
    if not token and not args.benchmark:
        sys.exit("AGENT_TOKEN is required (the indexer's FACADE_TOKEN)")
    indexer_url = os.getenv("INDEXER_URL", DEFAULT_INDEXER_URL)

    # an op missing on Apple GPU falls back to CPU instead of failing
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    log(f"Loading {MODEL_NAME}...")
    embedder = build_embedder(args)
    log(f"Model ready on {embedder.device_label}, batch {embedder.batch_size}")
    duty_cycle = getattr(embedder, "duty_cycle", None)
    if duty_cycle and (duty_cycle.default < 1 or duty_cycle.path):
        log(f"Duty cycle {duty_cycle.current():g}" + (f" (file {duty_cycle.path})" if duty_cycle.path else ""))

    if os.getenv("SKIP_REFERENCE_CHECK", "").lower() not in ("1", "true", "yes"):
        cosine = check_reference(embedder)
        log(f"Reference check passed: cosine {cosine:.5f} with FlagEmbedding")

    rate = measure_rate(embedder, args.benchmark or max(4, embedder.batch_size))
    log(f"Measured speed: {rate:.2f} chunks/s on 600-word chunks")
    if args.benchmark:
        return

    worker = os.getenv("WORKER_NAME") or f"{socket.gethostname()}-{embedder.device.replace(':', '')}"
    client = IndexerClient(indexer_url, token, worker)
    agent = Agent(client, embedder, float(os.getenv("TARGET_SECONDS", "120")), rate)

    def shutdown(signum, _frame):
        log(f"Signal {signum}: finishing the current batch and releasing leases...")
        agent.stop.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    info = client.info(agent.stop)
    if info.get("model") != MODEL_NAME or info.get("dimension") != EMBEDDING_DIMENSION:
        sys.exit(f"Indexer expects {info.get('model')} / {info.get('dimension')}, this agent computes "
                 f"{MODEL_NAME} / {EMBEDDING_DIMENSION}")
    log(f"Worker {worker} working for {indexer_url}")
    agent.run()
    log("Stopped")


if __name__ == "__main__":
    main()
