FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
CMD ["python", "-m", "src.io_bench.training", "--run-name", "demo-train", "--steps", "50", "--mode", "train"]