FROM python:3.11-bullseye

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
# CPU-only PyTorch for smaller image and faster startup
RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/requirements.txt

# Copy the entire project
COPY . /workspace/

# Make scripts executable
RUN chmod +x /workspace/start_sse.sh /workspace/agent_mock.py

# Expose web dashboard port
EXPOSE 8080

# Set environment variable to force CPU (can be overridden)
ENV FORCE_CPU=true
ENV PYTHONUNBUFFERED=1

# Keep container running by starting the agent
# This is the key command that keeps the container alive
CMD ["python3", "-u", "agent_sse.py"]
