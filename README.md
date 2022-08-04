# The NY Estimator Problem
In this challenge, we will explore the use of Airbnb listing data to predict the price category for new listings. We want to represent a real-case scenario where the MLE is working hand-to-hand with the Data Scientists at Intelygenz.

In this case, the data scientists have handed us a set of notebooks (in `lab/analysis`) that describe the ML workflow for data preprocessing and modelling. They have also included the dataset used and the trained model.

We will use these notebooks as a baseline to create more optimized functions that can be used in an ML inference pipeline.
# The MLE Challenge
You have to fork this repository to complete the following challenges in your own `github` account. Feel free to solve the challenge however you want.

Once completed, add a `SOLUTIONS.md` file justifying your responses and don't forget to send back the solution.

If you have any doubts or questions, don't hesitate to open an issue to ask any question about any challenge.

## Challenge 1 - Refactor DEV code

The code included in `lab` has been developed by Data Scientists during the development stage of the project. Now it is time to take their solution into production, and for that we need to ensure the code is up to standard and optimised. The first challenge is to refactor the code in `lab/analysis` the best possible way to operate in production.

Not only optimisation is important at this stage, but also the code should be written and tested in a way that can be easily understood by other MLE and tested at different CI stages.

## Challenge 2 - Build an API

The next step is to build an API that make use of the trained model to define the price category for a new listing. Here is an example of an input/output payload for the API.

```json
input = {
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
}

output = {
    "id": 1001,
    "price_category": "High"
}
```

The key is to ensure the API is easy to use and easy to test. Feel free to architect the API in any way you like and use any framework you feel comfortable with. Just ensure it is easy to make calls to the API in a local setting.

## Challenge 3 - Dockerize your solution

Nowadays, we can't think of ML solutions in production without thinking about Docker and its benefits in terms of standardisation, scalability and performance. The objective here is to dockerize your API and ensure it is easy to deploy and run in production.

## Solution

In this section I've included comments, reasons and decisions taken about ml-challenge

### Challenge-1
- I think that is convenient drop the column 'price' if we extract the category from this column (Talk with DS)
- I've created several config classes. This classes encapsulate config information and, depending of where will be deploy, have to be refactored for read this config file from different origins (environment variables, config files, etc)
- Amenities has been desglosed in several other columns, but not used in the inference process. I've tried to included these columns but the improve in the model is minimal. Should I include it? (Talk with DS)
- Could we categorize property_type to? (Talk with DS)
- I've included the metrics in the final pickle with the model, like dictionary. Maybe with this information could be interesting use some framework like ML Flow

### Challenge-2
- The api is as simple as possible and for time I haven't included enough testing on this part. These have to be the next steps

### Challenge-3
- I've only included the dockerfile and script for launch the api, not the code for running the container with this image. The instructions are as follows:
  ```
  docker build -t ml-challenge
  docker run --name ml-challenge-container -p 80:80 ml-challenge
  ```
