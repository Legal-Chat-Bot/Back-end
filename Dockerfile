FROM python:3.11-slim

WORKDIR /app

RUN pip install --root-user-action=ignore \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성 설치
RUN pip install --root-user-action=ignore -r requirements.txt

# kss는 PyYAML 충돌 때문에 deps 없이 설치
RUN pip install --root-user-action=ignore --no-deps kss==6.0.4

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]