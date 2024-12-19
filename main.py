from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

import os
import openai
from openai import OpenAI


def load_keys(path):
  with open(path, 'r') as file:
    return file.readline().strip()

def text2summary(input_text):
    client = OpenAI()
    system_role = '당신은 입력된 텍스트를 가장 간결하고 핵심적인 표현으로 요약하는 어시스턴트입니다. 불필요한 설명을 생략하고 핵심적인 정보를 짧은 문장으로 요약하세요.'

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_role
            },
            {
                "role": "user",
                "content": f'"{input_text}" 어떤 상황인지 알 수 있게 문장자체를 요약해줘'
            }
        ]
    )

    return response.choices[0].message.content


app = FastAPI()

path = './'

openai.api_key = load_keys(path + 'api_key.txt')
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
    
    summary = text2summary(request)
    
    return {"요약" : summary, "latitude" : latitude, "longitude" : longitude}

@app.get("/list_of_hospital")
def list_of_hospital():
    return {"병원1":"병원1"}, {"병원2":"병원2"}, {"병원3":"병원3"}