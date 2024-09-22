# Use the jupyter/pyspark-notebook image as a base image
FROM jupyter/pyspark-notebook:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PYTHONPATH=/app

# Install Java (required for Spark)
USER root
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
    apt-get clean

# Install additional Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Set the working directory
WORKDIR /app


# Expose the port for the FastAPI application
EXPOSE 8000

# Set the entrypoint
# Ensure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["api"]