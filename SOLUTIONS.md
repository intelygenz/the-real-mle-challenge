# MLE Challenge Solution

## Overview

This challenge involves using Airbnb listing data to predict the price category for new listings. The task is divided into three main challenges:

1. Refactor the development code for production.
2. Build an API to use the trained model for predictions.
3. Dockerize the solution for easy deployment and scalability.

I worked on solving this challenge on September 22nd, from 2:00 PM to 8:00 PM.

## Solution

I created five modules: data_preprocessing, training, inference, configuration and tests.
Appart from that I maintain the models and data folders.

To build the project:

```bash
docker build -t the_real_mle_challenge .
```

### Data Preprocessing

- Objective: Transform the exploratory Jupyter notebook code into a production script.
- Changes:
  - Migrated from pandas to PySpark to boost performance.
  - Separated transformations into auxiliary functions for better readability.

### Training

- Objective: Implement the logic to build the classifier.
- Changes:
  - Called the data_preprocessing module.
  - Performed additional data preprocessing steps specific to the model.
  - Trained the model, computed and logged metrics, and serialized the model using Pickle.
  - Refactored with the future goal of automating model retraining.
  - To run the training module:

```bash
docker run -it --rm the_real_mle_challenge train
```

### Inference

- Objective: Load the serialized model and perform inference based on API queries.
- Implemented the API using a framework (e.g., FastAPI) to ensure it is easy to use and test.
- The API receives input data, processes it, and returns the predicted price category.
- To run the inference module:

```bash
docker run -it --rm -p 8000:8000 the_real_mle_challenge
```

You can check the API documentation here: [http://localhost:8000/docs](http://localhost:8000/docs).

To make a request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "id": 1001,
           "accommodates": 4,
           "room_type": "Entire home/apt",
           "beds": 2,
           "bedrooms": 1,
           "bathrooms": 2,
           "neighbourhood": "Brooklyn",
           "tv": 1,
           "elevator": 1,
           "internet": 0,
           "latitude": 40.71383,
           "longitude": -73.9658
         }'
```

One gets this answer:

```bash
{"id":1001,"price_category":"High"}
```

### Tests

- Objective: Implement automatic tests to ensure the code is reliable and works as expected.
- Solution:
- Implemented tests for the data_preprocessing module.
- Used static data for input and expected output.
- Tested the loading function and the preprocessing output.
- To run the tests:

```bash
docker run -it --rm the_real_mle_challenge test
```

### Test Results

First test: Passed.
Second test: Failed with the following error:
```bash
DataFrame.iloc[:, 12] (column name="category") values are different (0.05025 %)
At positional index 8695, first diff: 0 != nan

----------------------------------------------------------------------
Ran 2 tests in 16.557s
```

This issue may be due to a mistake when converting from pandas to PySpark. Further debugging is required, but the tests demonstrate how they would work in production.

## Further Enhancements

- Split the modules into separate Docker containers.
- Build an ETL pipeline to feed upstream data and populate a relational database instead of writing .csv files or Spark partitions on disk.
- Implement batch and real-time processing (with Kafka) according to latency needs.