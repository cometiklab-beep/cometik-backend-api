# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala ffmpeg, necesario para el procesamiento de audio con Whisper
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Copiar e Instalar Dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el Codigo
COPY . .

# Exponer el Puerto de la API
EXPOSE 8000

# Comando de Ejecucion (Iniciar Uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
