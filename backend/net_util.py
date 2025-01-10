import os
from dotenv import load_dotenv
from openai import OpenAI
import json
from functools import lru_cache
from datetime import datetime, timedelta

# Load environment variables from .env
load_dotenv(dotenv_path='.env')

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
)

class QueryCache:
    def __init__(self, cache_file="json/query_cache.json", cache_duration_days=7):
        self.cache_file = cache_file
        self.cache_duration = timedelta(days=cache_duration_days)
        self.cache = self.load_cache()

    def load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
                # Filter out expired entries
                current_time = datetime.now()
                filtered_cache = {
                    k: v for k, v in cache_data.items()
                    if datetime.fromisoformat(v['timestamp']) + self.cache_duration > current_time
                }
                return filtered_cache
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def get(self, query):
        if query in self.cache:
            entry = self.cache[query]
            if datetime.fromisoformat(entry['timestamp']) + self.cache_duration > datetime.now():
                return entry['expanded_query']
        return None

    def set(self, query, expanded_query):
        self.cache[query] = {
            'expanded_query': expanded_query,
            'timestamp': datetime.now().isoformat()
        }
        self.save_cache()

# Usage in expand_query function
query_cache = QueryCache()

def expand_query(query):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant that specializes in ps2 video game titles. Return ONLY the 3-4 most relevant ps2 games, separated by commas."},
                {"role": "user",
                 "content": f"List 3 ps2 games as queries CLOSELY related to: {query}"}
            ],
            max_tokens=50,
            temperature=0.5
        )

        expanded_query = response.choices[0].message.content
        print("Expanded Query:", expanded_query)
        query_cache.set(query, expanded_query)
        return expanded_query

    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return query  # Return original query if expansion fails