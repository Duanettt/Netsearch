import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv(dotenv_path='.env')

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
)

def expand_query(query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that specializes in video game titles. When given a game title or series, respond with related games in a brief, concise way."},
                {"role": "user", "content": f" (Separate by commas) State all the games related to remember make these concise and short and also include the game company: {query}"}
            ],
            max_tokens=100,
            temperature=0.7
        )

        expanded_query = response.choices[0].message.content
        print("Expanded Query:", expanded_query)
        return expanded_query

    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return query  # Return original query if expansion fails