import os
import json
import requests
import re
import nltk
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup

stops = set(stopwords.words('english'))

# Define the path to your HTML files

# Static method for reading in our folder path for our files
def read_folder_path(user_folder_path):
    file_path_list = []
    for filename in os.listdir(user_folder_path):
            if filename.endswith('.html'):
                file_path = os.path.join(user_folder_path, filename)
                file_path_list.append(file_path)
                # print(file_path) # Print each file path to check it's looping correctly
    return file_path_list

# REGEX method: Needed to clean html for easier tokenization.
def cleanup_html(html):
    # Remove JavaScript blocks (simple approach, works for inline JS)
    text = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)  # Inline script tags
    text = re.sub(r'document.write\(.*?\);', '', text)  # Document writes
    text = re.sub(r'<[^>]+>', '', text)  # HTML tags
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # Inline comments (/*...*/)
    text = re.sub(r'//.*', '', text)  # Inline comments (//...)

    # Remove unnecessary punctuation, leftover quotes, etc.
    text = re.sub(r'["\'{}();]', '', text)  # Specific symbols
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing whitespace

    return text

def remove_stopwords(tokens):
    cleanedTokens = []
    cleanedTokens += [t for t in tokens if t not in stops]
    return cleanedTokens


class HTMLParser:
    def __init__(self, file_path_list):
        self.file_path_list = file_path_list
        self.parsed_html_list = []
        self.tokenized_data = []
        self.categorized_data = {}

    def parse_html(self):
        parsed_html_list = []  # Create a list to store parsed HTML content
        for file_path in self.file_path_list:
           with open(file_path, 'r', encoding='utf-8') as file:
               content = file.read()
               soup = BeautifulSoup(content, 'html5lib')
               parsed_html = soup.prettify()

               cleaned_html = cleanup_html(parsed_html)
               self.parsed_html_list.append(cleaned_html) # Return the soup object to allow us to prettify when we want to.

        return self.parsed_html_list # Return the list of parsed HTML content

    # def tokenize_html(self):
    #     for i in range(len(self.parsed_html_list)):
    #         tokenized_word = word_tokenize(self.parsed_html_list[i])
    #         self.tokenized_data.append({'file': os.path.basename(self.file_path_list[i]), 'tokens': tokenized_word})
    #     return self.tokenized_data

    def tokenize_content(self, cleaned_html):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cleaned_html)
        cleaned_tokens = remove_stopwords(tokens)
        return cleaned_tokens

    def categorize_content(self, title, file_path, cleaned_tokens):
        if title not in self.categorized_data:
            self.categorized_data[title] = []

        file_name = os.path.basename(file_path)

        # Append the file info and tokens under the title
        self.categorized_data[title].append({
            'file': file_name,
            'tokens': cleaned_tokens
        })

    def parse_and_process_html(self):
        print("Now parsing and processing HTML files!")
        for file_path in self.file_path_list:
            print(f"[INFO] Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    print(f"[INFO] Successfully read content from: {file_path}")

                    # Parse HTML content using BeautifulSoup
                    soup = BeautifulSoup(content, 'html5lib')

                    # Extract the title of the page
                    title = soup.title.string if soup.title else 'Title'
                    print(f"[DEBUG] Extracted title: {title}")

                    # Extract content with the specific ID
                    information = soup.find_all(id='content')
                    if not information:
                        print(f"[WARNING] No content found with ID 'content' in file: {file_path}")
                    information_contents = [info.get_text() for info in information]
                    print(f"[DEBUG] Extracted text from content ID: {information_contents}")

                    # Clean up HTML content
                    cleaned_html = cleanup_html(''.join(information_contents))
                    print(f"[DEBUG] Cleaned HTML content: {cleaned_html[:100]}...")

                    # Tokenize the cleaned HTML
                    tokens = self.tokenize_content(cleaned_html)
                    print(f"[DEBUG] Tokenized content ({len(tokens)} tokens): {tokens[:10]}...")

                    # Categorize content
                    self.categorize_content(title, file_path, tokens)
                    print(f"[INFO] Categorized content under title: {title}")

            except Exception as e:
                print(f"[ERROR] Failed to process file: {file_path}. Error: {e}")

    # Debug methods write outs.
    def parse_html_write_out(self):
        with open("parse_html_output.txt", 'w', encoding='utf-8') as file:
            for i in range(len(self.parsed_html_list)):
                file.write(self.parsed_html_list[i])

    def tokenise_html_write_out(self):
        with open("tokenise_output.txt", 'w', encoding='utf-8') as file:
            for i in range(len(self.parsed_html_list)):
                tokenized_word = word_tokenize(self.parsed_html_list[i])
                file.write(" ".join(tokenized_word) + "\n")

    def write_categorized_data_json(self, output_filename="categorized_data.json"):
        """Write the categorized data to a JSON file for structured storage."""
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(self.categorized_data, file, ensure_ascii=False, indent=4)

class WebScraper:
    def __init__(self, userURL):
        self.URL = userURL
        self.r = ''

    def make_request(self):
        self.r = requests.get(self.URL)
        soup = BeautifulSoup(self.r.content, 'html5lib')
        print(soup.prettify())

# Iterate over each file in the folder