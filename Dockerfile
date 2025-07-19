# Use a Python base image (e.g., Python 3.9, adjust if you use a different version)
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT: Pre-download Whisper model to bake it into the image ---
# This speeds up application startup and avoids re-downloading on every restart.
# Ensure XDG_CACHE_HOME is set to a path within the image where it can write.
ENV XDG_CACHE_HOME="/app/.cache"
RUN mkdir -p /app/.cache/whisper && \
    python -c "import whisper; whisper.load_model('base');"

# Copy the rest of your application code into the container
COPY . .

# Expose the port Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit application
# The `mkdir -p` ensures the persistent_storage directory exists inside the container
# before Streamlit tries to start, important if mounting fails or during initial runs.
# The ${PERSISTENT_STORAGE_PATH:-/app/persistent_storage} means use the env var if set,
# otherwise default to /app/persistent_storage.
CMD mkdir -p ${PERSISTENT_STORAGE_PATH:-/app/persistent_storage} && \
    streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false