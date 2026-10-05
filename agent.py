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

Pause and stop buttons on the indexer's vectorization page reach the agent with lease / results responses:
    PAUSE  finish the current batch, send it, give the other books back, unload the model (frees the GPU),
           keep asking the indexer; on resume load the model again (with the reference check) and go on
    STOP   finish the current batch, send it, give the other books back and exit (code 0)

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
    GPU_FAN_SPEED, GPU_POWER_LIMIT, GPU_TARGET_TEMP  quiet NVIDIA GPU while working (root): see embeddings/gpu_cooling.py
    EMBED_URL        use an OpenAI-compatible /embeddings server (TEI, infinity, LiteLLM) instead of local torch
    EMBED_MODEL      model name for EMBED_URL (default BAAI/bge-m3), EMBED_API_KEY its key
    SKIP_REFERENCE_CHECK=true  skip the startup comparison with the FlagEmbedding reference vector
"""

import argparse
import gc
import os
import queue
import signal
import socket
import sys
import threading
import time
import uuid
from typing import Callable, Dict, List, Optional

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
from embeddings.gpu_cooling import GpuCooling

VERSION = "2.2.0"
DEFAULT_INDEXER_URL = "https://book-indexer.svc.fred.org.ru"
SUBMIT_PART = 128          # chunks per POST /results (~700 KB of JSON)
MAX_CHUNKS_PER_LEASE = 2000


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class IndexerClient:
    def __init__(self, url: str, token: str, worker: str):
        self.url = url.rstrip("/") + "/api/agent/embeddings"
        self.worker = worker
        # new on every start: the indexer binds STOP to the process that received it
        self.session_id = uuid.uuid4().hex
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
            "session": self.session_id,
        }, timeout=120, stop=stop)

    def submit(self, results: List[dict], done: List[str], rate: float, stop: threading.Event) -> dict:
        return self._post("results", {
            "worker": self.worker,
            "model": MODEL_NAME,
            "results": results,
            "done": done,
            "chunksPerSec": round(rate, 3),
            "session": self.session_id,
        }, timeout=600, stop=stop)

    def release(self) -> None:
        try:
            response = self.session.post(f"{self.url}/release", json={"worker": self.worker}, timeout=30)
            if response.ok:
                log(f"Released {response.json().get('booksReleased', 0)} leased books")
        except requests.RequestException as e:
            log(f"Release failed ({e}); leases will expire on their own")


def free_device_memory() -> None:
    """Give cached GPU / Apple GPU memory back after the model is dropped."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception as e:  # noqa: BLE001 - best effort
            log(f"Could not free torch memory: {e}")
    if "mlx.core" in sys.modules:
        mx = sys.modules["mlx.core"]
        try:
            (getattr(mx, "clear_cache", None) or mx.metal.clear_cache)()
        except Exception as e:  # noqa: BLE001
            log(f"Could not free MLX memory: {e}")


class Agent:
    def __init__(self, client: IndexerClient, embedder, target_seconds: float, rate: float,
                 load_embedder: Optional[Callable[[int], object]] = None, cooling: Optional[GpuCooling] = None):
        self.client = client
        # fan / power limit of an NVIDIA card: taken over while working, given back during a pause
        self.cooling = cooling
        self.embedder = embedder
        # builds a ready embedder (model + reference check) again after a pause, given the batch size
        self.load_embedder = load_embedder
        self.device_label = embedder.device_label
        self.backend = embedder.backend
        self.target_seconds = target_seconds
        self.rate = rate  # chunks per second, exponential moving average of compute only
        self.stop = threading.Event()
        # PAUSE from the indexer: no new work, model unloaded until it is lifted
        self.paused = threading.Event()
        self.stopped_by_indexer = False
        # set only when shutdown gives up on sending computed results
        self.abort = threading.Event()
        self.leases: "queue.Queue[dict]" = queue.Queue(maxsize=1)
        self.results: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=2)
        self.total_chunks = 0
        self.started = time.time()

    def max_chunks(self) -> int:
        return int(min(MAX_CHUNKS_PER_LEASE, max(1, self.rate * self.target_seconds)))

    def on_command(self, command: Optional[str]) -> None:
        """Command from a lease or results response (buttons on the vectorization page)."""
        if command == "STOP":
            if not self.stop.is_set():
                log("Stop requested by the indexer: finishing the current batch, then exiting")
                self.stopped_by_indexer = True
                self.stop.set()
        elif command == "PAUSE":
            if not self.paused.is_set():
                log("Pause requested by the indexer: finishing the current batch, then freeing the device")
                self.paused.set()
        elif self.paused.is_set():
            log("Pause lifted by the indexer")
            self.paused.clear()

    # -- threads -----------------------------------------------------------

    def fetch_loop(self) -> None:
        """Keeps one lease ready while the model works on the current one."""
        while not self.stop.is_set():
            try:
                lease = self.client.lease(self.max_chunks(), self.rate, self.device_label,
                                          self.backend, self.stop)
            except ConnectionError:
                continue
            except ValueError as e:
                log(f"Lease failed: {e}")
                self.stop.wait(30)
                continue
            self.on_command(lease.get("command"))
            if lease.get("command"):
                self.stop.wait(lease.get("retryAfterSeconds") or 10)
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
            finally:
                self.results.task_done()

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
                    self.on_command(self.client.submit(part, done, self.rate, self.abort).get("command"))
                    part, done = [], []
            done.append(book["bookId"])
        if part or done:
            self.on_command(self.client.submit(part, done, self.rate, self.abort).get("command"))

    def pause(self) -> None:
        """Send what is computed, give the leased books back, unload the model and wait for the pause to end."""
        self.drop_prefetched()
        while self.results.unfinished_tasks and not self.stop.is_set():
            time.sleep(0.5)
        self.client.release()
        if self.cooling:
            self.cooling.pause()
        batch_size = self.embedder.batch_size
        if self.load_embedder is not None:
            self.embedder = None
            free_device_memory()
            log("Paused: model unloaded, device is free")
        else:
            log("Paused")
        while self.paused.is_set() and not self.stop.is_set():
            self.stop.wait(1)
        if self.stop.is_set():
            return
        if self.embedder is None:
            log(f"Resuming: loading {MODEL_NAME}...")
            self.embedder = self.load_embedder(batch_size)
            log(f"Model ready on {self.embedder.device_label}")
        if self.cooling:
            self.cooling.resume()
        self.drop_prefetched()

    def drop_prefetched(self) -> None:
        """A lease fetched before the pause: its books are released along with the rest."""
        try:
            while True:
                self.leases.get_nowait()
        except queue.Empty:
            pass

    # -- main --------------------------------------------------------------

    def run(self) -> None:
        fetcher = threading.Thread(target=self.fetch_loop, name="lease", daemon=True)
        submitter = threading.Thread(target=self.submit_loop, name="submit", daemon=True)
        fetcher.start()
        submitter.start()
        try:
            while not self.stop.is_set():
                if self.paused.is_set():
                    self.pause()
                    continue
                try:
                    lease = self.leases.get(timeout=1)
                except queue.Empty:
                    continue
                if self.paused.is_set():
                    continue  # released by pause() together with the other books
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

    # NVIDIA only, needs root: fixed fan, power limit, duty cycle following the temperature
    cooling = None if isinstance(embedder, HttpEmbedder) else GpuCooling.from_env(embedder.device)
    if cooling:
        cooling.start()
    try:
        check = os.getenv("SKIP_REFERENCE_CHECK", "").lower() not in ("1", "true", "yes")
        if check:
            cosine = check_reference(embedder)
            log(f"Reference check passed: cosine {cosine:.5f} with FlagEmbedding")

        def reload_embedder(batch_size: int):
            """After a pause: the same model on the same device, checked again; keeps the batch size it settled on."""
            fresh = build_embedder(args)
            fresh.batch_size = batch_size
            if check:
                log(f"Reference check passed: cosine {check_reference(fresh):.5f} with FlagEmbedding")
            return fresh

        rate = measure_rate(embedder, args.benchmark or max(4, embedder.batch_size))
        log(f"Measured speed: {rate:.2f} chunks/s on 600-word chunks")
        if args.benchmark:
            return

        worker = os.getenv("WORKER_NAME") or f"{socket.gethostname()}-{embedder.device.replace(':', '')}"
        client = IndexerClient(indexer_url, token, worker)
        agent = Agent(client, embedder, float(os.getenv("TARGET_SECONDS", "120")), rate,
                      load_embedder=None if isinstance(embedder, HttpEmbedder) else reload_embedder, cooling=cooling)
        if cooling:
            cooling.duty_cycle = lambda: getattr(agent.embedder, "duty_cycle", None)
        # the agent owns the model now: no other reference may keep it in memory during a pause
        del embedder, duty_cycle

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
        log("Stopped by the indexer" if agent.stopped_by_indexer else "Stopped")
    finally:
        if cooling:
            cooling.stop()


if __name__ == "__main__":
    main()
