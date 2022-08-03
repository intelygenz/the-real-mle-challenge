"""
This file contains the definition of the api, as simple as possible, to make an inference from model
"""

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
