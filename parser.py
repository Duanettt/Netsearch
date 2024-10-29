import os
import json
from bs4 import BeautifulSoup

# Define the path to your HTML files
folder_path = './videogames/'
class HTMLParser:
    def __init__(self, html_path):
        self.path = html_path
        self.file_path_list = []
        self.parsed_html_list = []

    def parse_html(self):
        parsed_html_list = []  # Create a list to store parsed HTML content
        self.file_path_list.clear()
        for filename in os.listdir(self.path):
            if filename.endswith('.html'):
                file_path = os.path.join(self.path, filename)
                self.file_path_list.append(file_path)
                  # Print each file path to check it's looping correctly

                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                soup = BeautifulSoup(content, 'html5lib')
                parsed_html = soup.prettify()
                self.parsed_html_list.append(parsed_html) # Return the soup object to allow us to prettify when we want to.

        return self.parsed_html_list # Return the list of parsed HTML content

    def parse_html_write(self):
        parsed_html = self.parse_html()

        with open("output.txt", 'w', encoding='utf-8') as file:
            file.write(json.dumps(parsed_html))



# Iterate over each file in the folder