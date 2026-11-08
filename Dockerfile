FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and built-in templates
COPY llmproxy/ /app/llmproxy/
COPY templates/ /app/templates/
COPY plugins/ /app/plugins/
COPY assets/ /app/assets/
COPY README.md /app/README.md

# Ensure runtime directories exist
RUN mkdir -p /app/data /app/plugins /app/templates

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "llmproxy.main:app", "--host", "0.0.0.0", "--port", "8000"]
