#!/usr/bin/env python3
"""
Embedding Agent for Library Search System - RS HTTP Facade Version

Consumes text chunks from rs-http-facade via SSE, computes embeddings using BGE-M3,
and submits results back via HTTP facade.

Uses limit=1 parameter on SSE connect to process one message at a time.

IMPORTANT: Uses shared embedding module from common-py to ensure consistency.
"""

import argparse
import os
import sys
import time
import uuid
import json
import requests
import threading
from typing import Optional, Dict, Any, List
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from flask import Flask, render_template, jsonify

# Import from local embedding module
from embeddings import load_model, compute_embedding, MODEL_NAME, EMBEDDING_DIMENSION


class EmbeddingAgentSSE:
    """Embedding agent that uses SSE to consume from rs-http-facade with limit=1"""
    
    def __init__(self, facade_url: str, token: str, task_stream: str, task_group: str, result_stream: str, device: Optional[str] = None):
        self.facade_url = facade_url.rstrip('/')
        self.token = token
        self.task_stream = task_stream
        self.task_group = task_group
        self.result_stream = result_stream
        self.agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        # Consumer name for the new finish API - generated once and reused
        # Use full UUID to ensure uniqueness across distributed deployments
        self.consumer_name = f"consumer-{uuid.uuid4()}"
        
        # HTTP session for publishing results
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        })
        
        # Statistics
        self.start_time = datetime.now()
        self.stats = {
            'tasks_processed': 0,
            'compute_times': [],
            'publish_times': [],
            'connected': False,
            'last_message_time': None,
            'connection_attempts': 0
        }
        self.stats_lock = threading.Lock()  # Thread-safe access to stats

        # Model will be loaded separately (after Flask starts)
        self.model = None
        self.device = device

    def load_embedding_model(self):
        """Load the BGE-M3 model (call this after Flask server is started)"""
        print(f"[{self.agent_id}] Loading BGE-M3 model from shared module...")
        if self.device:
            print(f"[{self.agent_id}] Forcing device: {self.device}")
        self.model = load_model(device=self.device)
        print(f"[{self.agent_id}] Model loaded successfully")
        print(f"[{self.agent_id}] Model: {MODEL_NAME}, Dimension: {EMBEDDING_DIMENSION}")
            
    def compute_embedding(self, text: str) -> tuple[list[float], float]:
        """Compute embedding for text. Returns (embedding, compute_time)"""
        compute_start = time.time()
        with open(os.devnull, 'w') as devnull, redirect_stderr(devnull), redirect_stdout(devnull):
            embedding = compute_embedding(self.model, text)
        compute_time = time.time() - compute_start
        return embedding, compute_time
        
    def publish_result(self, message_id: str, task: Dict[str, Any], embedding: List[float]) -> tuple[bool, float]:
        """Publish result via HTTP facade with retry. Returns (success, publish_time)"""
        url = f"{self.facade_url}/api/streams/{self.result_stream}/messages"

        # Both task and result use camelCase for consistency
        # Include taskMessageId so backend can send FINISH for task queue (V2 pipeline)
        result = {
            'chunkUid': task['chunkUid'],
            'bookId': task['bookId'],
            'chunkIndex': task['chunkIndex'],
            'sqlitePath': task['sqlitePath'],
            'localChunkId': task['localChunkId'],
            'modelName': MODEL_NAME,
            'dim': EMBEDDING_DIMENSION,
            'dtype': 'float32',
            'embedding': embedding,
            'taskMessageId': message_id  # V2: Backend will send FINISH for this task
        }

        payload = {'data': result}

        # Retry with exponential backoff
        max_retries = 5
        backoff = 2  # seconds

        for attempt in range(max_retries):
            try:
                publish_start = time.time()
                response = self.session.post(url, json=payload, timeout=30)
                publish_time = time.time() - publish_start

                if response.status_code in (200, 201):
                    return (True, publish_time)
                else:
                    print(f"[{self.agent_id}] Failed to publish result: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"[{self.agent_id}] Retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    return (False, publish_time)
            except requests.exceptions.ConnectionError as e:
                publish_time = time.time() - publish_start if 'publish_start' in locals() else 0
                if attempt < max_retries - 1:
                    print(f"[{self.agent_id}] Connection error publishing result (facade might be down), retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    print(f"[{self.agent_id}] Connection error after {max_retries} attempts: {e}")
                    return (False, publish_time)
            except Exception as e:
                publish_time = time.time() - publish_start if 'publish_start' in locals() else 0
                print(f"[{self.agent_id}] Failed to publish result: {e}")
                return (False, publish_time)

        return (False, 0)
            
    def finish_message(self, message_id: str) -> bool:
        """Finish (ACK) a message with retry
        Uses the new API: POST /api/streams/{stream}/groups/{group}/consumers/{consumer}/messages/{messageId}/finish
        """
        url = f"{self.facade_url}/api/streams/{self.task_stream}/groups/{self.task_group}/consumers/{self.consumer_name}/messages/{message_id}/finish"

        # Retry with exponential backoff
        max_retries = 5
        backoff = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = self.session.post(url, timeout=10)
                if response.status_code == 200:
                    return True
                elif response.status_code == 404:
                    # Message not found - already processed or topic deleted
                    print(f"[{self.agent_id}] Message {message_id[:16]}... not found (already processed or topic deleted)")
                    return True  # Treat as success - message is gone anyway
                else:
                    print(f"[{self.agent_id}] Failed to finish message: HTTP {response.status_code}")
                    if attempt < max_retries - 1:
                        print(f"[{self.agent_id}] Retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    return False
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    print(f"[{self.agent_id}] Connection error finishing message (facade might be down), retrying in {backoff}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                else:
                    print(f"[{self.agent_id}] Connection error after {max_retries} attempts, treating as success (message will be redelivered if needed)")
                    return True  # After retries exhausted, treat as success to not block
            except Exception as e:
                print(f"[{self.agent_id}] Failed to finish message {message_id}: {e}")
                return False

        return False
            
    def process_message(self, sse_message: Dict[str, Any]) -> bool:
        """Process a single SSE message. Returns True if successful."""
        message_id = sse_message['id']

        try:
            # Get task from message body (already parsed as dict)
            task = sse_message['body']

            # Get text directly from task message
            text = task.get('text')
            if not text:
                print(f"[{self.agent_id}] No text field found in task message, will be redelivered after timeout")
                return False
                
            # Compute embedding
            embedding, compute_time = self.compute_embedding(text)

            # Publish result (includes taskMessageId for V2 backend to send FINISH)
            success, publish_time = self.publish_result(message_id, task, embedding)

            # Update stats (thread-safe)
            with self.stats_lock:
                self.stats['compute_times'].append(compute_time)
                self.stats['publish_times'].append(publish_time)
                self.stats['last_message_time'] = datetime.now()

            if success:
                # Send FINISH to allow next message delivery immediately
                # No need to wait for backend processing - that's asynchronous
                finish_success = self.finish_message(message_id)

                with self.stats_lock:
                    self.stats['tasks_processed'] += 1
                total_time = compute_time + publish_time
                print(f"[{self.agent_id}] ✓ Task {task['chunkUid'][:16]}... completed | "
                      f"Compute={compute_time:.3f}s | Publish={publish_time:.3f}s | Total={total_time:.3f}s")

                # Print stats every 10 tasks
                if self.stats['tasks_processed'] % 10 == 0:
                    self._print_stats()
                return finish_success
            else:
                # Failed to publish - will be redelivered after timeout
                print(f"[{self.agent_id}] Failed to publish result, will be redelivered after timeout")
                return False
                
        except Exception as e:
            print(f"[{self.agent_id}] Error processing message: {e}, will be redelivered after timeout")
            return False
            
    def _print_stats(self):
        """Print timing statistics"""
        if self.stats['tasks_processed'] == 0:
            return
            
        def avg(times):
            return sum(times[-10:]) / min(len(times), 10) if times else 0
            
        compute_avg = avg(self.stats['compute_times'])
        publish_avg = avg(self.stats['publish_times'])
        
        print(f"[{self.agent_id}] 📊 Stats (last 10 tasks): "
              f"Compute={compute_avg:.3f}s | Publish={publish_avg:.3f}s | "
              f"Total processed={self.stats['tasks_processed']}")
              
    def run(self):
        """Main agent loop - consumes messages via SSE and processes them one at a time"""
        # Use limit=1 parameter to control in-flight messages (replaces RDY mechanism)
        url = f"{self.facade_url}/api/events?stream={self.task_stream}&group={self.task_group}&consumer={self.consumer_name}&limit=1"
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'text/event-stream'
        }
        
        print(f"[{self.agent_id}] Agent started, consuming from {self.task_stream}/{self.task_group}")
        print(f"[{self.agent_id}] Consumer name: {self.consumer_name}")
        print(f"[{self.agent_id}] Using limit=1 (one message at a time)")
        
        try:
            while True:
                # Track connection attempt
                with self.stats_lock:
                    self.stats['connection_attempts'] += 1
                    self.stats['connected'] = False

                try:
                    # Timeout 120s - server sends keep-alive comments every 60s
                    with requests.get(url, headers=headers, stream=True, timeout=120) as response:
                        if response.status_code != 200:
                            print(f"[{self.agent_id}] Failed to connect: HTTP {response.status_code} - {response.text}")
                            time.sleep(5)
                            continue

                        # Mark as connected
                        with self.stats_lock:
                            self.stats['connected'] = True

                        print(f"[{self.agent_id}] Connected to SSE stream with limit=1")

                        # Add heartbeat logging
                        last_heartbeat = time.time()
                        message_count = 0

                        # Buffer for reading SSE stream properly
                        buffer = ""

                        for raw_chunk in response.iter_content(chunk_size=4096, decode_unicode=False):
                            if not raw_chunk:
                                continue

                            # Facade omits charset in Content-Type, so decode raw bytes as UTF-8 explicitly
                            chunk = raw_chunk.decode('utf-8', errors='replace')
                            buffer += chunk

                            # Process all complete events in buffer (separated by \n\n)
                            while '\n\n' in buffer:
                                    # Heartbeat logging every 30 seconds
                                    now = time.time()
                                    if now - last_heartbeat > 30:
                                        print(f"[{self.agent_id}] Heartbeat: Still connected, processed {message_count} messages so far")
                                        last_heartbeat = now

                                    # Extract one complete event
                                    event, buffer = buffer.split('\n\n', 1)
                                    event = event.strip()

                                    if event.startswith('data:'):
                                        data = event[5:].strip()
                                        if data:
                                            try:
                                                message = json.loads(data)
                                                message_count += 1
                                                print(f"[{self.agent_id}] Received message #{message_count}, ID={message.get('id', 'unknown')}")
                                                # Process message immediately (synchronous, no queue)
                                                # After finish, server will automatically send next message
                                                self.process_message(message)
                                            except json.JSONDecodeError as e:
                                                print(f"[{self.agent_id}] Failed to parse SSE message: {e}")
                                                print(f"[{self.agent_id}] Data was: {data[:200]}...")
                                    elif event and not event.startswith(':'):
                                        # Log other non-comment SSE events for debugging
                                        print(f"[{self.agent_id}] SSE event: {event[:100]}")

                        # Connection closed
                        with self.stats_lock:
                            self.stats['connected'] = False
                        print(f"[{self.agent_id}] SSE connection closed, reconnecting...")

                except requests.exceptions.Timeout as e:
                    with self.stats_lock:
                        self.stats['connected'] = False
                    print(f"[{self.agent_id}] SSE stream timeout (no data received for 120s, expected keep-alive every 60s)")
                    print(f"[{self.agent_id}] Reconnecting...")
                    time.sleep(2)
                except requests.exceptions.ConnectionError as e:
                    with self.stats_lock:
                        self.stats['connected'] = False
                    print(f"[{self.agent_id}] Connection error (topic might be deleted or service restarting): {e}")
                    print(f"[{self.agent_id}] Waiting 10 seconds before reconnecting...")
                    time.sleep(10)
                except requests.exceptions.RequestException as e:
                    with self.stats_lock:
                        self.stats['connected'] = False
                    print(f"[{self.agent_id}] Request error: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print(f"\n[{self.agent_id}] Shutting down...")

    def get_stats_snapshot(self) -> Dict[str, Any]:
        """Get a thread-safe snapshot of current statistics"""
        with self.stats_lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            compute_times = self.stats['compute_times'][-100:]  # Last 100
            publish_times = self.stats['publish_times'][-100:]  # Last 100

            def avg(times):
                return sum(times) / len(times) if times else 0

            return {
                'agent_id': self.agent_id,
                'uptime_seconds': int(uptime),
                'connected': self.stats['connected'],
                'tasks_processed': self.stats['tasks_processed'],
                'last_message_time': self.stats['last_message_time'].isoformat() if self.stats['last_message_time'] else None,
                'connection_attempts': self.stats['connection_attempts'],
                'avg_compute_time': round(avg(compute_times), 3),
                'avg_publish_time': round(avg(publish_times), 3),
                'model_name': MODEL_NAME,
                'embedding_dimension': EMBEDDING_DIMENSION,
                'facade_url': self.facade_url,
                'task_stream': self.task_stream,
                'task_group': self.task_group
            }


def create_flask_app(agent: EmbeddingAgentSSE) -> Flask:
    """Create Flask web application for statistics dashboard"""
    app = Flask(__name__)

    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('dashboard.html')

    @app.route('/api/stats')
    def stats():
        """API endpoint for statistics"""
        return jsonify(agent.get_stats_snapshot())

    return app


def run_flask_server(app: Flask, port: int):
    """Run Flask server (to be run in a separate thread)"""
    # Disable Flask's startup messages
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Embedding Agent for Library Search System (SSE Mode)'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage even if CUDA or MPS is available'
    )
    args = parser.parse_args()

    # Read configuration from environment variables
    # Hardcoded configuration values - these never change
    facade_url = 'https://nsq.fred.org.ru'
    task_stream = 'embedding_tasks'
    task_group = 'embedding-agent'
    result_stream = 'embedding_results'
    
    # Only configurable parameter
    token = os.getenv('RS_HTTP_FACADE_TOKEN')
    
    # Determine device: command-line flag takes precedence, then environment variable
    force_cpu = args.cpu or os.getenv('FORCE_CPU', '').lower() in ('true', '1', 'yes')
    device = 'cpu' if force_cpu else None

    # Web dashboard port - hardcoded
    web_port = 8080

    if not token:
        print("Error: RS_HTTP_FACADE_TOKEN environment variable is required")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("Embedding Agent Configuration:")
    print(f"  Facade URL:    {facade_url}")
    print(f"  Task Stream:   {task_stream}")
    print(f"  Task Group:    {task_group}")
    print(f"  Result Stream: {result_stream}")
    print(f"  Token:         {'*' * 20}{token[-4:] if len(token) > 4 else '****'}")
    print(f"  Force CPU:     {force_cpu}")
    print(f"  Web Dashboard: http://0.0.0.0:{web_port}")
    print("=" * 60)

    # Create agent (without loading model yet)
    agent = EmbeddingAgentSSE(facade_url, token, task_stream, task_group, result_stream, device=device)

    # Start Flask web server FIRST (available immediately)
    flask_app = create_flask_app(agent)
    flask_thread = threading.Thread(target=run_flask_server, args=(flask_app, web_port), daemon=True)
    flask_thread.start()
    print(f"[{agent.agent_id}] Web dashboard started at http://0.0.0.0:{web_port}")
    print(f"[{agent.agent_id}] Dashboard is available while model loads...")

    # NOW load the model (this takes 1-2 minutes, but dashboard is already available)
    agent.load_embedding_model()

    # Run agent (blocking)
    agent.run()


if __name__ == '__main__':
    main()
