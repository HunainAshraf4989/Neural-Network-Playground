# Backend image for the deployed HF Docker Space (deploy plan S0/S1).
# CPU-only torch via the dedicated index — the default CUDA build balloons the
# image by ~8 GB and the Space has no GPU anyway.
FROM python:3.12-slim

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# HF Spaces run containers as UID 1000; the repo dir is not writable by it,
# which is why server mode defaults LOG_FILE to "" (stderr-only, see config.py).
RUN useradd -m -u 1000 user
USER user

ENV MODE=server \
    PORT=7860 \
    LOG_FILE=

EXPOSE 7860

CMD ["python", "backend/main.py"]
