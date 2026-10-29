FROM python:3.11-slim

# Evita a criação de arquivos temporários .pyc e força flush imediato de stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODELS_DIR=/app/models

WORKDIR /app

# Instala bibliotecas de sistema necessárias para OpenCV, GL e monitoramento
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalação e cache otimizado das dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Cópia estruturada do código-fonte e modelos pré-treinados
COPY src/ /app/src/
COPY models/ /app/models/

EXPOSE 8000

# Execução do servidor assíncrono ASGI via Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

