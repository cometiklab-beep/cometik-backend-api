#!/usr/bin/env bash

# Ejecuta el comando de inicio robusto
# Esto es más seguro que solo el Procfile en algunos entornos.
/opt/render/project/src/.venv/bin/python -m gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
