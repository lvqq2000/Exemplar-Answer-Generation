import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def generate_answer(prompt):
    try:
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    while True:
        question = input("Enter your question: ")
        answer = generate_answer(question)
        if answer:
            print("Generated exemplar Answer:", answer)