from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import os
import openai
from openai import OpenAI
import sys

import json

import emergency_ai26 as em

app = FastAPI()

path = './'

openai.api_key = em.load_keys(path + 'api_key.txt')
os.environ['OPENAI_API_KEY'] = openai.api_key

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello":"World"}


@app.get("/hospital_by_module")
def hospital_by_module(request : str, latitude : float, longitude : float):
    
    summary = em.text2summary(request)
    
    return {"요약" : summary, "latitude" : latitude, "longitude" : longitude}