FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src PIP_NO_CACHE_DIR=1
WORKDIR /app

# dépendances d'abord (meilleur cache de build)
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY models ./models
COPY assets ./assets
COPY data ./data
COPY .streamlit ./.streamlit

# données ouvertes téléchargées à la construction, empreintes vérifiées
RUN python scripts/download_data.py

RUN useradd -m rag && chown -R rag /app
USER rag

EXPOSE 8501 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)" || exit 1

# Démo Streamlit (défaut) :   docker run -p 8501:8501 rag-ecc
# API REST :                  docker run -p 8000:8000 rag-ecc uvicorn app.api:app --host 0.0.0.0 --port 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
