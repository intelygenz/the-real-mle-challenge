from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from app.models import ModelInput, ModelOutput
from app.utils import load_model, load_transformer

FEATURE_NAMES = ['neighbourhood', 'room_type', 'accommodates', 'bathrooms', 'bedrooms']
OUT_CLASSES = np.array(['Low', 'Mid', 'High', 'Lux'])

app = FastAPI(
    title= "Building Category prediction",
    description= "Api to infer the price category of a building from its characteristics",
    version= "1.0.0", 
    docs_url="/"
)


@app.post("/model-inference")
async def infer_price_caegory(input: ModelInput):

    model = load_model()
    transformer = load_transformer()
    
    model_input = dict(input)
    
    if model:
        try:  
            # build data frame with the input to the transformer
            # if all the field dont have the same lenght it will raise an error
            if isinstance(model_input['id'], int):
                input_data = pd.DataFrame(model_input, index=[0])
            else:
                input_data = pd.DataFrame(model_input, index=list(range(len(model_input['id']))))
            
            # preprocess the data 
            data = transformer.predict(input_data)
            data = data[FEATURE_NAMES].dropna(axis=0)
            category = model.predict(data)
            # parse the numerical outpout to the corresponding classes
            category_str = OUT_CLASSES[category]

            return ModelOutput(id=input.id, price_category=category_str[0] if len(category) == 1 else category_str)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during the prediction: {str(e)}")
    else:
        return HTTPException(status_code=500, detail="Model or pipeline not ready")
