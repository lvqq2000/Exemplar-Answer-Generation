import os
from openai import OpenAI

from config import OPENAI_API_KEY, MODEL_NAME

def generate_answer(prompt):
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None
    
