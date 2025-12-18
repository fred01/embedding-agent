FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
# CPU-only PyTorch for smaller image and faster startup
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the entire project
COPY . /workspace/

# Make scripts executable
RUN chmod +x /workspace/start_sse.sh /workspace/agent_mock.py

# Expose web dashboard port
EXPOSE 5000

# Set environment variable to force CPU (can be overridden)
ENV FORCE_CPU=true
ENV PYTHONUNBUFFERED=1

# Keep container running by starting the agent
# This is the key command that keeps the container alive
CMD ["python3", "-u", "agent_sse.py"]
