#!/usr/bin/env python3
"""
Mock Embedding Agent for Testing Queue Infrastructure

Consumes text chunks from rs-http-facade via SSE, generates random embeddings,
and submits results back via HTTP facade.

Uses RDY=1 mechanism to process one message at a time - no local buffering.
Simplified version without requeue logic for pure queue testing.
"""

import os
import sys
import time
import uuid
import json
import random
import requests
from typing import Dict, Any, List


# Mock embedding configuration
MODEL_NAME = "mock/random-v1"
EMBEDDING_DIMENSION = 1024


class MockEmbeddingAgent:
    """Mock agent that generates random embeddings for queue testing"""

    def __init__(self, facade_url: str, token: str, task_topic: str, task_channel: str, result_topic: str):
        self.facade_url = facade_url.rstrip('/')
        self.token = token
        self.task_topic = task_topic
        self.task_channel = task_channel
        self.result_topic = result_topic
        self.agent_id = f"mock-agent-{uuid.uuid4().hex[:8]}"

        # HTTP session for publishing results and RDY control
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        })

        # Statistics
        self.stats = {
            'tasks_processed': 0,
            'compute_times': [],
            'publish_times': []
        }

        # Track if RDY has been set after connection
        self.rdy_initialized = False

        print(f"[{self.agent_id}] Mock agent initialized")
        print(f"[{self.agent_id}] Model: {MODEL_NAME}, Dimension: {EMBEDDING_DIMENSION}")

    def set_rdy(self, count: int = 1, retry: bool = False, max_retries: int = 10):
        """Set RDY count for the consumer"""
        url = f"{self.facade_url}/api/consumers/{self.task_topic}/{self.task_channel}/rdy"
        payload = {'count': count}

        for attempt in range(max_retries):
            try:
                response = self.session.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    return True
                elif response.status_code == 404 and retry:
                    # Consumer not yet created, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    else:
                        return False
                else:
                    return False
            except requests.exceptions.ConnectionError:
                # Connection refused - topic/channel might be deleted, this is normal during cleanup
                return False
            except Exception:
                if retry and attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return False
        return False

    def compute_embedding(self, text: str) -> tuple[list[float], float]:
        """Generate random embedding. Returns (embedding, compute_time)"""
        compute_start = time.time()

        # Simulate real computation time (3-4 seconds)
        time.sleep(random.uniform(3.0, 4.0))

        # Generate random vector normalized to unit length
        embedding = [random.gauss(0, 1) for _ in range(EMBEDDING_DIMENSION)]
        # Normalize to unit length (like real embeddings)
        magnitude = sum(x*x for x in embedding) ** 0.5
        embedding = [x / magnitude for x in embedding]

        compute_time = time.time() - compute_start
        return embedding, compute_time

    def publish_result(self, message_id: str, task: Dict[str, Any], embedding: List[float]) -> tuple[bool, float]:
        """Publish result via HTTP facade. Returns (success, publish_time)"""
        url = f"{self.facade_url}/api/streams/{self.result_topic}/messages"

        # Both task and result use camelCase for consistency
        result = {
            'chunkUid': task['chunkUid'],
            'bookId': task['bookId'],
            'chunkIndex': task['chunkIndex'],
            'sqlitePath': task['sqlitePath'],
            'localChunkId': task['localChunkId'],
            'modelName': MODEL_NAME,
            'dim': EMBEDDING_DIMENSION,
            'dtype': 'float32',
            'embedding': embedding
        }

        payload = {'data': result}

        try:
            publish_start = time.time()
            response = self.session.post(url, json=payload, timeout=30)
            publish_time = time.time() - publish_start

            if response.status_code in (200, 201):
                return (True, publish_time)
            else:
                print(f"[{self.agent_id}] Failed to publish result: HTTP {response.status_code}")
                return (False, publish_time)
        except Exception as e:
            publish_time = time.time() - publish_start if 'publish_start' in locals() else 0
            print(f"[{self.agent_id}] Failed to publish result: {e}")
            return (False, publish_time)

    def finish_message(self, message_id: str) -> bool:
        """Finish (ACK) a message"""
        url = f"{self.facade_url}/api/messages/{message_id}/finish"

        try:
            response = self.session.post(url, timeout=10)
            if response.status_code == 200:
                return True
            elif response.status_code == 404:
                # Message not found - already processed or topic deleted
                return True  # Treat as success - message is gone anyway
            else:
                print(f"[{self.agent_id}] Failed to finish message: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"[{self.agent_id}] Failed to finish message {message_id}: {e}")
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
                print(f"[{self.agent_id}] No text field found in task message, skipping")
                # Just finish the message without processing
                self.finish_message(message_id)
                return False

            # Generate mock embedding
            embedding, compute_time = self.compute_embedding(text)
            self.stats['compute_times'].append(compute_time)

            # Publish result
            success, publish_time = self.publish_result(message_id, task, embedding)
            self.stats['publish_times'].append(publish_time)

            if success:
                # Finish message
                if self.finish_message(message_id):
                    self.stats['tasks_processed'] += 1
                    total_time = compute_time + publish_time
                    print(f"[{self.agent_id}] ✓ Task {task['chunkUid'][:16]}... completed | "
                          f"Compute={compute_time:.3f}s | Publish={publish_time:.3f}s | Total={total_time:.3f}s")

                    # Print stats every 10 tasks
                    if self.stats['tasks_processed'] % 10 == 0:
                        self._print_stats()
                    return True
                else:
                    # Failed to finish - will be redelivered
                    print(f"[{self.agent_id}] Failed to finish message, will be redelivered")
                    return False
            else:
                # Failed to publish - just finish anyway to avoid infinite loop
                print(f"[{self.agent_id}] Failed to publish, finishing message to avoid redelivery")
                self.finish_message(message_id)
                return False

        except Exception as e:
            print(f"[{self.agent_id}] Error processing message: {e}")
            # Finish message to avoid infinite loop
            self.finish_message(message_id)
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
        url = f"{self.facade_url}/api/events?stream={self.task_topic}&group={self.task_channel}"
        headers = {
            'Authorization': f'Bearer {self.token}',
            'Accept': 'text/event-stream'
        }

        print(f"[{self.agent_id}] Mock agent started, consuming from {self.task_topic}/{self.task_channel}")
        print(f"[{self.agent_id}] Using RDY=1 (one message at a time)")

        try:
            while True:
                try:
                    with requests.get(url, headers=headers, stream=True, timeout=None) as response:
                        if response.status_code != 200:
                            print(f"[{self.agent_id}] Failed to connect: HTTP {response.status_code}")
                            time.sleep(5)
                            continue

                        print(f"[{self.agent_id}] Connected to SSE stream")

                        # Set RDY=1 after connection is established (with retry logic)
                        if not self.rdy_initialized:
                            print(f"[{self.agent_id}] Waiting for consumer to be created...")
                            if self.set_rdy(1, retry=True, max_retries=10):
                                self.rdy_initialized = True
                                print(f"[{self.agent_id}] RDY initialized, waiting for messages...")
                            else:
                                print(f"[{self.agent_id}] Failed to initialize RDY, reconnecting...")
                                continue

                        # Add heartbeat logging
                        last_heartbeat = time.time()
                        message_count = 0

                        # Buffer for reading SSE stream properly
                        buffer = ""

                        for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
                            if chunk:
                                buffer += chunk

                                # Process all complete events in buffer (separated by \n\n)
                                while '\n\n' in buffer:
                                    # Heartbeat logging every 30 seconds
                                    now = time.time()
                                    if now - last_heartbeat > 30:
                                        print(f"[{self.agent_id}] Heartbeat: Still connected, processed {message_count} messages")
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
                                                print(f"[{self.agent_id}] Received message #{message_count}, ID={message.get('id', 'unknown')[:16]}...")
                                                # Process message immediately (synchronous, no queue)
                                                self.process_message(message)
                                                # After processing, set RDY=1 again to get next message
                                                self.set_rdy(1)
                                            except json.JSONDecodeError as e:
                                                print(f"[{self.agent_id}] Failed to parse SSE message: {e}")

                        print(f"[{self.agent_id}] SSE connection closed, reconnecting...")
                        self.rdy_initialized = False  # Reset flag on disconnect

                except requests.exceptions.ConnectionError as e:
                    print(f"[{self.agent_id}] Connection error: {e}")
                    self.rdy_initialized = False  # Reset RDY flag
                    print(f"[{self.agent_id}] Waiting 10 seconds before reconnecting...")
                    time.sleep(10)
                except requests.exceptions.RequestException as e:
                    print(f"[{self.agent_id}] Request error: {e}")
                    self.rdy_initialized = False  # Reset RDY flag
                    time.sleep(5)

        except KeyboardInterrupt:
            print(f"\n[{self.agent_id}] Shutting down...")


def main():
    # Read configuration from environment variables
    # Hardcoded configuration values - these never change
    facade_url = 'https://nsq.fred.org.ru'
    task_topic = 'embedding_tasks'
    task_channel = 'embedding-agent'
    result_topic = 'embedding_results'
    
    # Only configurable parameter
    token = os.getenv('RS_HTTP_FACADE_TOKEN')

    if not token:
        print("Error: RS_HTTP_FACADE_TOKEN environment variable is required")
        sys.exit(1)

    # Print configuration
    print("=" * 60)
    print("Mock Embedding Agent Configuration:")
    print(f"  Facade URL:    {facade_url}")
    print(f"  Task Topic:    {task_topic}")
    print(f"  Task Channel:  {task_channel}")
    print(f"  Result Topic:  {result_topic}")
    print(f"  Model:         {MODEL_NAME}")
    print(f"  Dimension:     {EMBEDDING_DIMENSION}")
    print(f"  Token:         {'*' * 20}{token[-4:] if len(token) > 4 else '****'}")
    print("=" * 60)

    # Create and run agent
    agent = MockEmbeddingAgent(facade_url, token, task_topic, task_channel, result_topic)
    agent.run()


if __name__ == '__main__':
    main()
