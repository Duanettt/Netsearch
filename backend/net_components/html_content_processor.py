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
from nltk.stem import PorterStemmer, WordNetLemmatizer
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
        self.categorized_data = {}
        self.parsed_content_list = []
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # condition variables
        self.use_json = False
        self.use_stemming = False
        self.use_lemmatizer = False




    # def tokenize_html(self):
    #     for i in range(len(self.parsed_html_list)):
    #         tokenized_word = word_tokenize(self.parsed_html_list[i])
    #         self.tokenized_data.append({'file': os.path.basename(self.file_path_list[i]), 'tokens': tokenized_word})
    #     return self.tokenized_data

    def parse_and_process_html(self):
        print("Now parsing and processing HTML files!")
        # condition variables to allow for loading json.
        if self.use_json:
            try:
                self.load_from_json()
                print("[INFO] Loaded data from JSON file.")
                return
            except FileNotFoundError:
                print("[INFO] JSON file not found. Proceeding to parse HTML.")

        # reading from file path
        for file_path in self.file_path_list:
            print(f"[INFO] Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    print(f"[INFO] Successfully read content from: {file_path}")

                    # we parse using beautifulsoup library and html5lib features.
                    soup = BeautifulSoup(content, 'html5lib')

                    # extract the url from each file path to use as a result
                    url = f"file://{os.path.abspath(file_path)}"

                    # extract the title
                    title = soup.title.string if soup.title else 'Title'
                    print(f"[DEBUG] Extracted title: {title}")

                    # extract content id div.
                    information = soup.find_all(id='content')
                    if not information:
                        print(f"[WARNING] No content found with ID 'content' in file: {file_path}")
                    information_contents = [info.get_text() for info in information]
                    print(f"[DEBUG] Extracted text from content ID: {information_contents}")

                    # Extract game bio information
                    game_bio_info = {}
                    bio_table = soup.find('table', class_='gameBioInfo')

                    # Within the html files theres always a table with gameBioInfo as a class we can iterate through these
                    # and obtain some metadata for extra weights.

                    if bio_table:
                        rows = bio_table.find_all('tr')
                        for row in rows:
                            header = row.find('td', class_='gameBioInfoHeader')
                            text = row.find('td', class_='gameBioInfoText')

                            if header and text:
                                header_text = header.get_text(strip=True)
                                info_text = text.get_text(strip=True)
                                game_bio_info[header_text] = info_text

                    # Clean up HTML content
                    # cleaned_html = cleanup_html(''.join(information_contents))
                    # print(f"[DEBUG] Cleaned HTML content: {cleaned_html[:100]}...")

                    # Tokenize the cleaned HTML
                    tokens = self.tokenize_content(''.join(information_contents))
                    print(f"[DEBUG] Tokenized content ({len(tokens)} tokens): {tokens[:10]}...")

                    # text processing involves using stemming or using lemmatisation.
                    tokens = self.process_text(tokens)

                    # Categorize content
                    self.categorize_content(title, file_path, tokens, url, game_bio_info)
                    print(f"[INFO] Categorized content under title: {title}")

            except Exception as e:
                print(f"[ERROR] Failed to process file: {file_path}. Error: {e}")

            self.save_to_json()

    def tokenize_content(self, cleaned_html):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(cleaned_html)
        cleaned_tokens = remove_stopwords(tokens)
        return tokens

    def categorize_content(self, title, file_path, cleaned_tokens, url, game_bio_info=None):
        if title not in self.categorized_data:
            self.categorized_data[title] = []

        file_name = os.path.basename(file_path)

        # Create entry with game bio info
        entry = {
            'file': file_name,
            'tokens': cleaned_tokens,
            'url': url.replace('\\', '/'),
            'game_info': game_bio_info or {}  # Include game bio info if available
        }

        # Using the title add the entry so we can identify the document.
        self.categorized_data[title].append(entry)


    def save_to_json(self, output_filename="json/parsed_data.json"):
        """Save parsed data to a JSON file."""
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump({
                'categorized_data': self.categorized_data
            }, file, ensure_ascii=False, indent=4)
        print(f"[INFO] Data successfully saved to {output_filename}")

    def load_from_json(self, input_filename="json/parsed_data.json"):
        """Load parsed data from a JSON file."""
        try:
            with open(input_filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
                self.categorized_data = data['categorized_data']
            print(f"[INFO] Data successfully loaded from {input_filename}")
        except FileNotFoundError:
            print(f"[ERROR] JSON file {input_filename} not found.")
        except Exception as e:
            print(f"[ERROR] Failed to load data from JSON file: {e}")




    def set_use_stemming(self, enabled : bool):
        self.use_stemming = enabled

    def set_use_lemmatizer(self, enabled : bool):
        self.use_lemmatizer = enabled

    def set_use_json(self, enabled : bool):
        self.use_json = enabled

    def process_text(self, tokens):
        """Process text with optional stemming and lemmatization"""
        # Apply stemming and/or lemmatization
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        if self.use_lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        return tokens
    # Debug methods/test methods
    '''
    ________________________________________________________________________
    '''
    def parse_html_write_out(self):
        with open("parse_html_output.txt", 'w', encoding='utf-8') as file:
            for i in range(len(self.parsed_html_list)):
                file.write(self.parsed_html_list[i])

    def tokenise_html_write_out(self):
        with open("tokenise_output.txt", 'w', encoding='utf-8') as file:
            for i in range(len(self.parsed_html_list)):
                tokenized_word = word_tokenize(self.parsed_html_list[i])
                file.write(" ".join(tokenized_word) + "\n")

    def write_categorized_data_json(self, output_filename="/json/categorized_data.json"):
        """Write the categorized data to a JSON file for structured storage."""
        with open(output_filename, 'w', encoding='utf-8') as file:
            json.dump(self.categorized_data, file, ensure_ascii=False, indent=4)

    def parse_html(self):
        parsed_html_list = []  # Create a list to store parsed HTML content
        for file_path in self.file_path_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html5lib')
                parsed_html = soup.prettify()

                cleaned_html = cleanup_html(parsed_html)
                self.parsed_html_list.append(
                    cleaned_html)  # Return the soup object to allow us to prettify when we want to.

        return self.parsed_html_list  # Return the list of parsed HTML content

        # adding a parse content to test if only parsing the content div gives better results:

    def parse_content(self):
        """Parse only the content div from HTML files"""
        parsed_content_list = []
        for file_path in self.file_path_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                soup = BeautifulSoup(content, 'html5lib')

                # Find the content div specifically
                content_div = soup.find('div', id='content')
                if content_div:
                    # Extract just the text from the content div
                    content_text = content_div.get_text(strip=True)
                    # Clean the extracted content
                    cleaned_content = cleanup_html(content_text)
                    parsed_content_list.append(cleaned_content)
                else:
                    print(f"Warning: No content div found in {file_path}")
                    parsed_content_list.append("")

        self.parsed_content_list = parsed_content_list
        return self.parsed_content_list

    def parse_and_process_html_content(self, use_json=False):
        print("Now parsing and processing HTML files!")
        # condition variable to allow for json loadng form current directory
        if use_json:
            try:
                self.load_from_json()
                print("[INFO] Loaded data from JSON file.")
                return
            except FileNotFoundError:
                print("[INFO] JSON file not found. Proceeding to parse HTML.")

        for file_path in self.file_path_list:
            print(f"[INFO] Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    soup = BeautifulSoup(content, 'html5lib')

                    url = f"file://{os.path.abspath(file_path)}"
                    title = soup.title.string if soup.title else 'Title'

                    # Extract only the content div
                    content_div = soup.find('div', id='content')
                    if content_div:
                        # Get text from content div
                        content_text = content_div.get_text(strip=True)
                        gameBioInfo = soup.find('table', id='gameBioInfo')
                        cleaned_content = cleanup_html(content_text)
                        tokens = self.tokenize_content(cleaned_content)
                        self.categorize_content(title, file_path, tokens, url)
                        print(f"[INFO] Successfully processed content from: {file_path}")
                    else:
                        print(f"[WARNING] No content div found in file: {file_path}")

            except Exception as e:
                print(f"[ERROR] Failed to process file: {file_path}. Error: {e}")

        self.save_to_json()
