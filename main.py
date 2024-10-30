import parser
from parser import HTMLParser
from parser import WebScraper
from parser import read_folder_path

folder_path = './videogames/'
file_path_list = read_folder_path(folder_path)

user_parser = HTMLParser(file_path_list)

parsed_html_list = user_parser.parse_html()
user_parser.tokenise_html_write_out()
# print(parsed_html_list[1])

