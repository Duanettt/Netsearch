import os
import json
import requests
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from bs4 import BeautifulSoup

# Define the path to your HTML files

# Static method for reading in our folder path for our files
def read_folder_path(user_folder_path):
    file_path_list = []
    for filename in os.listdir(user_folder_path):
            if filename.endswith('.html'):
                file_path = os.path.join(user_folder_path, filename)
                file_path_list.append(file_path)
                print(file_path) # Print each file path to check it's looping correctly
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

    def tokenize_html(self):
        for i in range(len(self.parsed_html_list)):
            tokenized_word = word_tokenize(self.parsed_html_list[i])
            self.tokenized_data.append({'file': os.path.basename(self.file_path_list[i]), 'tokens': tokenized_word})
        return self.tokenized_data

    def parse_html_categorize_tc(self):
        for file_path in self.file_path_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html5lib')

                # So firstly we base our tokens on our title so we find our titles

                title = soup.title.string if soup.title else 'Title'
                # We then find our information based on the content ID in our div tag in the website.
                information = soup.find_all(id='content')

                information_contents = [info.get_text() for info in information]
                ## We then clean our html up by removing tags and spaces etcetera
                cleaned_html = cleanup_html(''.join(information_contents))

                # Tokenize cleaned HTML content
                # We created our own tokenizer solely based on words.
                tokenizer = RegexpTokenizer(r'\w+')

                # We then ,using this tokenizer, to tokenize based on the regex expressed above
                tokens = tokenizer.tokenize(cleaned_html)

                 # We then for each entry in our dictionary have to check if the title is in there or not to prevent errors
                if title not in self.categorized_data:
                    self.categorized_data[title] = []

                # Append the file info and tokens as a new entry under the main heading
                self.categorized_data[title].append({
                    'file': os.path.basename(file_path),
                    'title': title,
                    'information': cleaned_html,
                    'tokens': tokens
                })


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

# Figuring out this method
    # def write_categorized_data_txt(self, output_filename="categorized_data.txt"):
    #     with open(output_filename, 'w', encoding='utf-8') as file:
    #         for categorized_data in self.categorized_data.values():
    #




class WebScraper:
    def __init__(self, userURL):
        self.URL = userURL
        self.r = ''

    def make_request(self):
        self.r = requests.get(self.URL)
        soup = BeautifulSoup(self.r.content, 'html5lib')
        print(soup.prettify())

# Iterate over each file in the folder