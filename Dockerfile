# 1. Base Image: Use an official Python image.
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies
# ffmpeg is required by openai-whisper for audio processing.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg

# 4. Copy requirements file and install Python packages
# This leverages Docker's layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code into the container
COPY . .

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. Define the command to run your application
# This command starts the Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]