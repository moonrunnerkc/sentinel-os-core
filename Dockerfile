FROM python:3.12-slim

# Author: Bradley R. Kinnard
# sentinel-os-core container

# install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# optional: install firejail for enhanced sandbox
RUN apt-get update && apt-get install -y firejail || true \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# run static analysis during build
RUN pip install flake8 mypy bandit
RUN flake8 --max-line-length=100 --ignore=E501,W503 core/ memory/ security/ interfaces/ graphs/ utils/ || true
RUN mypy --ignore-missing-imports core/ memory/ security/ interfaces/ graphs/ utils/ || true
RUN bandit -r core/ memory/ security/ interfaces/ graphs/ utils/ -ll || true

# create data directories
RUN mkdir -p data/beliefs data/episodes data/logs data/models

# run tests
RUN python -m pytest tests/ -v --ignore=tests/benchmarks.py -m "not slow and not chaos" || true

# default command
CMD ["python", "main.py"]
