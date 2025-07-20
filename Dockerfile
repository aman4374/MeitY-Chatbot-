# Use a slim Python base image
FROM python:3.10-slim-buster

# Set environment variable for Streamlit to suppress telemetry (optional)
ENV STREAMLIT_HOME=/app \
    XDG_CACHE_HOME="/app/.cache" \
    PYTHONUNBUFFERED=1 \
    PERSISTENT_STORAGE_PATH="/app/persistent_storage"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload Whisper base model and create necessary cache dirs
RUN mkdir -p /app/.cache/whisper && \
    python -c "import whisper; whisper.load_model('base')"

# Copy application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Create persistent storage directory at runtime if it doesn't exist and run Streamlit
CMD mkdir -p "$PERSISTENT_STORAGE_PATH" && \
    streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
