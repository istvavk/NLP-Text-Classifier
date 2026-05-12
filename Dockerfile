FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch
RUN pip install --no-cache-dir numpy && \
    pip install --no-cache-dir \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY . .

ENTRYPOINT ["python", "textClassifier/cli.py"]
