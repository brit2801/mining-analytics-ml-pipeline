# Use an official Python image as the base image.
# The slim version is lighter and enough for running the FastAPI API.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and force logs to appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Copy dependency file first to improve Docker layer caching.
COPY requirements.txt .

# Install Python dependencies.
# --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the full project into the container.
COPY . .

# Expose the port used by FastAPI/Uvicorn.
EXPOSE 8000

# Start the FastAPI application.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]