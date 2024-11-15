# Use an official Python image as the base
FROM python:3.10-slim

# Set environment variables to avoid Python buffer issues
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Add /app and /app/api to PYTHONPATH
ENV PYTHONPATH=/app:/app/api

# Copy only requirements.txt first to leverage Docker layer caching
COPY requirements.txt /app/

# Install production dependencies only
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code
COPY api /app/api
COPY models /models

# Expose port 8000 to access FastAPI
EXPOSE 8000

# Command to run the API using Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]