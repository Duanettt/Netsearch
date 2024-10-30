import html_content_processor
from Netsearch.data_formatter import TokensDataStorage
from html_content_processor import HTMLParser
from html_content_processor import WebScraper
from html_content_processor import read_folder_path

import data_formatter

folder_path = './videogames/'
file_path_list = read_folder_path(folder_path)

user_parser = HTMLParser(file_path_list)

user_parser.parse_html_categorize_tc()

user_parser.write_categorized_data_json()

# print(parsed_html_list[1])

