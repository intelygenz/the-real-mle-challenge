# Property Price Prediction API

This project provides a FastAPI-based machine learning API for predicting property price categories based on various features of a listing. The API classifies listings into categories such as Low, Mid, High, and Luxury. 
The application is containerized using Docker to simplify deployment.

## Prerequisites

- **Docker**: Make sure Docker is installed on your machine. You can download it from [https://www.docker.com/get-started](https://www.docker.com/get-started).

## Running the API with Docker

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/polalbacar/the-real-mle-challenge.git
cd the-real-mle-challenge
```

### 2. Build the Docker Image

Build the Docker image using the following command. This will create an image named `ny-estimator-api` (you can change the name if desired):

```bash
docker build -t ny-estimator-api .
```

### 3. Run the Docker Container

Run the container with:

```bash
docker run -d -p 8000:8000 ny-estimator-api
```

- `-d` runs the container in detached mode.
- `-p 8000:8000` maps port 8000 on your machine to port 8000 in the container, where the FastAPI application is running.

### 4. Access the API

Once the container is running, you can access the API:

- **Swagger UI** (Interactive Documentation): [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

These links provide an interactive interface to explore and test the API endpoints.

### Stopping the Container

To stop the Docker container, use:

```bash
docker stop $(docker ps -q --filter ancestor=ny-estimator-api)
```

Or if you know the container ID:

```bash
docker stop <container_id>
```

## TLDR

- **Build** the image with `docker build -t ny-estimator-api .`
- **Run** the container with `docker run -d -p 8000:8000 ny-estimator-api`
- **Access** the API documentation at [http://localhost:8000/docs](http://localhost:8000/docs)