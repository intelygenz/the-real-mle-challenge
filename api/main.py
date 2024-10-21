from api.api import create_app
from api.estimators.ny_classifier import NYClassifier

MODEL_PATH = "./models/ny_classifier.pkl"

# Keep the classifier in memory to avoid loading it every time
classifier = NYClassifier.from_pickle(MODEL_PATH)

app = create_app(classifier)
