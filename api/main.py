from fastapi import FastAPI

from api.estimators.ny_classifier import NYClassifier
from api.schemas.listing import ListingSchema
from api.schemas.prediction import PredictionOutputSchema

MODEL_PATH = "./models/ny_classifier.pkl"


def create_app(classifier: NYClassifier) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def health_check():
        return {"status": "ok"}

    @app.post("/estimate", response_model=PredictionOutputSchema)
    def estimate(listing: ListingSchema):
        """Estimate the price category of a listing based on its features."""

        price_category = classifier.predict(listing)

        return {"id": listing.id, "price_category": price_category}

    return app


# Keep the classifier in memory to avoid loading it every time
classifier = NYClassifier.from_pickle(MODEL_PATH)

app = create_app(classifier)
