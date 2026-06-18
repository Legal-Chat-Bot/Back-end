# FROM python:3.11-slim

# WORKDIR /app

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1
# ENV PIP_NO_CACHE_DIR=1

# RUN apt-get update \
#     && apt-get install -y --no-install-recommends \
#         libgl1 \
#         libglib2.0-0 \
#         libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# RUN python -m pip install --upgrade pip setuptools wheel

# RUN pip install --root-user-action=ignore -r requirements.txt

# RUN pip install --root-user-action=ignore --no-deps kss==6.0.4

# COPY . .

# EXPOSE 8000

# CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

COPY . .

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]