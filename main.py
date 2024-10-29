import parser
from parser import HTMLParser

folder_path = './videogames/'
user_parser = HTMLParser(folder_path)

user_parsed_html_list = []
parsed_html_list = user_parser.parse_html()

test_parsed_html_str = parsed_html_list[1]
print(test_parsed_html_str)

