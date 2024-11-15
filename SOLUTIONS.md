# SOLUTIONS.md

This document explains the solutions implemented for the Inteligenz technical assessment, detailing the approach, methodology, and justifications for each of the three challenges.

## Challenge 1 - Refactor DEV Code

### Objective
The objective of this challenge was to take exploratory data science code from Jupyter notebooks and refactor it into a production-ready structure. This required converting it into modular, reusable, and optimized Python scripts that meet production standards.

### Solution

- I split the notebook code into separate scripts within the `scripts` directory, each with a purpose:
    - `data_loader.py`: Handles loading raw and preprocessed data.
    - `preprocessing.py`: Contains functions for cleaning and transforming data.
    - `model.py`: Contains functions for training, evaluating, saving, and loading the model.
    - `utils.py`: Provides utility functions, such as plotting and feature extraction.
    - `main.py`: Serves as the main script, running the complete ML pipeline, from data loading and preprocessing to training and evaluation.

- The `main.py` file is the central script and the only file necessary to run the entire pipeline. It:
     - Loads raw Airbnb listings data.
     - Preprocesses the raw data and saves the cleaned dataset.
     - Loads the preprocessed data and splits it for training and testing.
     - Trains the model using the processed data.
     - Evaluates the model and saves the evaluation results, including feature importance and confusion matrix.
     - Saves the trained model in a timestamped folder under `models`, along with its evaluation results. This setup ensures that each trained model is saved with its associated evaluation, allowing for comparisons between models. By keeping a history of models and their performance, we can identify and select the best-performing model.


- Each script has clear docstrings and comments to ensure that other Machine Learning Engineers (MLEs) can easily understand the code and purpose of each function.
- The code follows the PEP-8 style guide for readability, and functions are logically organized by responsibility. This modular approach makes it easier to test and maintain each part independently.
- I structured the code to facilitate testing, ensuring that each function performs a single, well-defined task. This makes it easy to test each component in isolation.

---

## Challenge 2 - Build an API

### Objective
The goal was to build an API using FastAPI to serve the trained model and classify a property listing based on input features. The API needed to be user-friendly, easy to test, and straightforward to call locally.

### Solution

- I chose **FastAPI** due to its speed, ease of use, and automatic interactive documentation with Swagger UI. FastAPI is also well-suited for serving machine learning models and offers simple request validation with Pydantic.
- The API exposes a `/predict` endpoint that accepts JSON input with features such as `id`, `accommodates`, `room_type`, and `neighbourhood`.
- The API returns a JSON output with the `id` and the predicted `price_category`, which is one of `Low`, `Mid`, `High`, or `Luxury`.
- Input validation is handled with Pydantic, ensuring that required fields are present and correctly formatted.
- The API preprocesses incoming data to match the format expected by the model. This includes encoding categorical variables (e.g., `room_type` and `neighbourhood`), which are mapped to integers as per the training data.
- The API is easy to use locally, and the Swagger UI, accessible at `/docs`, allows for interactive testing.

---

## Challenge 3 - Dockerize Your Solution

### Objective
The objective was to Dockerize the API for ease of deployment and scalability. Docker ensures a consistent environment, making it easier to deploy the application in different environments with the same setup.

### Solution

- I created a **Dockerfile** that defines the environment needed to run the API. The Dockerfile uses a lightweight `python:3.10-slim` base image to minimize the image size and it is enough for our case.
- The Dockerfile is structured to use Docker layer caching by copying only `requirements.txt` first, installing dependencies, and then copying the application code.
- The Docker container exposes port 8000, which maps to the FastAPI application’s port.
- A `CMD` command is used to start the FastAPI application with Uvicorn, listening on all interfaces to make it accessible outside the container.

3. **Usage Instructions**:
   - I included detailed instructions in `Read HowToDeploy.md` on building the Docker image and running the container.
   - Users can easily build the image with `docker build -t property-price-api .` and run it with `docker run -d -p 8000:8000 property-price-api`.

4. **Testing the Dockerized API**:
   - Once running, users can access the API documentation at `http://localhost:8000/docs` and test the endpoint.

---