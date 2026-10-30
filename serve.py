import os
from fastapi import FastAPI
from pydantic import BaseModel,Field
from model import Predictor

def create_app():
    predictor=Predictor(os.getenv('MODEL_DIR','artifacts'))
    app=FastAPI(title='Persian HVAC symptom classifier',version='0.1.0')
    class Input(BaseModel):
        text: str=Field(min_length=1,max_length=2000)
    @app.get('/health')
    def health():return {'status':'ok'}
    @app.post('/predict')
    def predict(body:Input):return predictor.predict(body.text)
    return app
