# Embedding Agent (SSE Mode)

This is the SSE-based version of the embedding agent that consumes messages directly from rs-http-facade.

## Overview

The SSE agent connects to rs-http-facade via Server-Sent Events (SSE) to consume chunk processing tasks, computes embeddings using BGE-M3, and publishes results back via HTTP.

## Key Differences from Legacy Agent

- **No polling**: Uses SSE for real-time message streaming instead of polling the indexer API
- **Direct queue interaction**: Communicates with rs-http-facade instead of the indexer API
- **Autonomous message handling**: Manages message lifecycle (finish/requeue) directly with the queue
- **Self-contained messages**: Chunk text is included in messages - no external dependencies

## Requirements

Same as the legacy agent:
- Python 3.8+
- ~5 GB free space for BGE-M3 model
- ML accelerator (optional but recommended): Apple Silicon M1/M2/M3, or NVIDIA GPU with CUDA
- Use `--cpu` flag or `FORCE_CPU=true` environment variable to run on CPU when accelerator is present but not desired

## Installation and Usage

### Quick Start

```bash
# Set required environment variable
export RS_HTTP_FACADE_TOKEN="your-secret-token"

# Run the agent directly
python3 agent_sse.py
```

Or use Docker (recommended):

```bash
docker run -e RS_HTTP_FACADE_TOKEN="your-secret-token" ghcr.io/fred01/embedding-agent:latest
```

### Environment Variables

- **`RS_HTTP_FACADE_TOKEN`** (required) - Authentication token for rs-http-facade

**Note**: The following configuration values are now hardcoded in the agent and cannot be changed:
- **Facade URL**: `https://nsq.fred.org.ru`
- **Task Stream**: `embedding_tasks`
- **Task Group**: `embedding-agent`
- **Result Stream**: `embedding_results`
- **Web Port**: `8080`

Optional:
- **`BATCH_SIZE`** (optional) - Number of messages to retrieve from SSE at once (default: `1`). Higher values reduce connection overhead when processing on GPU. Messages are still processed and finished one by one for stability. If the queue becomes empty and no new messages arrive within 5 seconds, buffered messages are processed immediately.
- **`FORCE_CPU`** (optional) - Force CPU usage even if CUDA/MPS is available (default: `false`). Set to `true`, `1`, or `yes` to enable.

### Command-Line Arguments

- **`--cpu`** - Force CPU usage even if CUDA or MPS is available. Equivalent to setting `FORCE_CPU=true`.

```bash
# Force CPU via command-line flag
python3 agent_sse.py --cpu

# Or via environment variable
export FORCE_CPU=true
python3 agent_sse.py
```

### Docker Compose Setup

The rs-http-facade service is automatically included in `docker-compose.yml`:

```yaml
services:
  rs-http-facade:
    image: ghcr.io/fred01/rs-http-facade:latest
    environment:
      BEARER_TOKEN: ${RS_HTTP_FACADE_TOKEN}
      PORT: 8090
    ports:
      - "8090:8090"
```

## How It Works

1. **SSE Connection**: Agent opens an SSE connection to `/api/events?stream={stream}&group={group}`
2. **Consumer Creation**: HTTP facade automatically creates a consumer when the SSE connection is established
3. **RDY Initialization**: After connection, agent sets RDY=1 (with retry logic) to receive one message at a time
4. **Receive Messages**: Messages are streamed in real-time as SSE events with chunk text included
5. **Compute Embedding**: BGE-M3 model computes the embedding vector from the text in the message
6. **Publish Result**: Result is published to result stream via HTTP POST
7. **Finish Message**: Original message is finished (ACKed) via HTTP POST to `/api/messages/{id}/finish`
8. **Reset RDY**: After processing, agent sets RDY=1 again to get the next message

## Architecture

```
┌─────────────┐       HTTP POST      ┌──────────────────┐
│   Indexer   │ ──────────────────> │  rs-http-facade  │
└─────────────┘   Enqueue chunks    │                  │
                                      │   ┌──────────┐  │
                                      │   │  Queue   │  │
                                      │   └──────────┘  │
                                      └────────┬─────────┘
                                              │ SSE Stream
                                              ▼
                                      ┌─────────────────┐
                                      │  Embedding      │
                                      │  Agent (SSE)    │
                                      │                 │
                                      │  ┌───────────┐  │
                                      │  │   BGE-M3  │  │
                                      │  └───────────┘  │
                                      └────────┬────────┘
                                              │ HTTP POST
                                              ▼
                                      ┌──────────────────┐
                                      │  rs-http-facade  │
                                      │  (Publish result)│
                                      └──────────────────┘
```

## Message Format

### Task Message (received via SSE)

```json
{
  "id": "message-id-from-queue",
  "timestamp": 1234567890,
  "attempts": 1,
  "body": {"chunkUid":"...","bookId":123,"chunkIndex":0,"text":"The actual chunk text content...","sqlitePath":"/path/to/shard.db","localChunkId":42}
}
```

### Result Message (published to stream)

```json
{
  "data": {
    "chunkUid": "abc123...",
    "bookId": 123,
    "chunkIndex": 0,
    "sqlitePath": "/path/to/shard.db",
    "localChunkId": 42,
    "modelName": "BAAI/bge-m3",
    "dim": 1024,
    "dtype": "float32",
    "embedding": [0.123, -0.456, ...]
  }
}
```

## Performance

Same performance characteristics as the legacy agent for embedding computation. The SSE-based approach provides:

- **Lower latency**: Real-time message delivery instead of polling
- **Better scalability**: Native load balancing across multiple consumers
- **Resource efficiency**: No wasted polling cycles

## Troubleshooting

### Connection errors

If you see SSE connection errors:
1. Verify rs-http-facade is running: `curl http://localhost:8090/admin/ping`
2. Check authentication token matches
3. Ensure the queue backend is accessible to rs-http-facade

### RDY initialization errors

If you see "Failed to set RDY: HTTP 404 - No consumers found":
- **This is expected on first connection** - the agent will automatically retry
- Consumer is created asynchronously when SSE connection is established
- Agent will retry setting RDY up to 10 times with 2-second delays
- If retries fail, check that SSE connection is actually established
- Verify rs-http-facade can communicate with the queue backend

### Message redelivery

If messages are being redelivered:
- Check that finish/requeue calls are succeeding
- Verify network connectivity to rs-http-facade
- Check rs-http-facade logs for errors
- Verify messages contain valid text field

## Monitoring

Check agent health:
- Watch console output for error messages
- Monitor task processing rate in logs
- Check queue depth: `curl http://localhost:8090/admin/stats?format=json`

## Migration from Legacy Agent

The SSE agent is a drop-in replacement for the legacy polling agent:

1. Stop the old agent
2. Ensure rs-http-facade is deployed at `https://nsq.fred.org.ru`
3. Update environment variable to `RS_HTTP_FACADE_TOKEN`
4. Run the agent with `python3 agent_sse.py` or use Docker

Both agents can run simultaneously during migration.
