FROM public.ecr.aws/docker/library/python:3.13-bookworm

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

# CPU-only PyTorch for a small image; GPU hosts: see Dockerfile.cuda in the repo root
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r /tmp/requirements.txt

COPY . /workspace/

ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-u", "agent.py"]
