# RecoForge 运行镜像（CPU，确定性可复现）
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app

# 先装依赖层（利用缓存）
COPY requirements.txt requirements-sota.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-sota.txt || true

COPY . .

# 默认运行端到端演示（确定性，落盘 benchmark.json）
CMD ["python", "-m", "recoforge.examples.run_demo", "--seed", "42", "--out", "benchmark.json"]
