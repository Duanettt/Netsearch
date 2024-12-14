import json

import html_content_processor
from data_formatter import TokensDataStorage
from html_content_processor import HTMLParser
from html_content_processor import WebScraper
from html_content_processor import read_folder_path

import data_formatter
from search_engine import SearchEngine

folder_path = './videogames/'

file_path_list = read_folder_path(folder_path)

user_parser = HTMLParser(file_path_list)

user_parser.parse_and_process_html()

search_engine = SearchEngine(user_parser)

search_engine.build_inverted_index()
print(search_engine.find_unique_terms())

search_engine.debug_tf_idf()

search_engine.user_prompt_tfidf()

search_engine.write_inverted_index_to_file()

# user_parser.write_categorized_data_json()

# print(parsed_html_list[1])

