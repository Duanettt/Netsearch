import json
import os

import html_content_processor
from data_formatter import TokensDataStorage
from html_content_processor import HTMLParser
from html_content_processor import WebScraper
from html_content_processor import read_folder_path

import data_formatter

class SearchEngine:
    def __init__(self, html_parser):
        self.HTMLParser = html_parser
        self.tokenized_data = []
        self.inverted_index = {}
        self.vocab = {}
        self.docIDs = {}
        self.postings = {}

        self.vocab_counter = 0   # Tracks unique ID for each vocab term
        self.doc_counter = 0     # Tracks unique ID for each document

    def build_inverted_index(self):
        categorized_data = self.HTMLParser.categorized_data
        for title, files in categorized_data.items():
            for file in files:
                file_name = file['file']
                tokens = file['tokens']
                self.update_inverted_index(file_name, tokens)

    def update_inverted_index(self, file_name, tokens):
        """
        Updates the inverted index with tokens and their file occurrences.
        """
        # Assign a unique ID to the document if it hasn't been processed yet
        if file_name not in self.docIDs:
            self.docIDs[file_name] = self.doc_counter
            docID = self.doc_counter
            self.doc_counter += 1
        else:
            docID = self.docIDs[file_name]

        # Process each token
        for token in tokens:
            # Assign a unique vocab ID to the token if it's not in the vocab
            if token not in self.vocab:
                self.vocab[token] = self.vocab_counter
                vocabID = self.vocab_counter
                self.vocab_counter += 1
            else:
                vocabID = self.vocab[token]

            # Update postings: add docID to the list for this vocabID
            if vocabID not in self.postings:
                self.postings[vocabID] = {docID}
            else:
                self.postings[vocabID].add(docID)

            ## Checking the term frequency
            token = token.lower()

            if token not in self.inverted_index:
                self.inverted_index[token] = {}

            if file_name not in self.inverted_index[token]:
                self.inverted_index[token][file_name] = 1
            else:
                self.inverted_index[token][file_name] += 1


    def finalize_postings(self):
        # Convert each set in postings to a sorted list for JSON compatibility
        for vocabID in self.postings:
            self.postings[vocabID] = sorted(self.postings[vocabID])

    def write_inverted_index_to_file(self):
        # Ensure postings are finalized before saving
        self.finalize_postings()
        with open("inverted_index.json", 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=4)


    def user_prompt(self):
        # We create a reverse look up document to allow us identify which html files match
        # Easier for debugging
        reverse_lookup_document_id = {v:k for k, v in self.docIDs.items()}
        # We then now have our user input which checks the query within our vocab matches the vocab
        # to our vocab ID and we use that for documents.
        while True:
            query = input("What would you like to search for? ")
            if query == "quit":
                break
            if query in self.vocab:
                vocabID = self.vocab[query]
                documents = self.postings[vocabID]
                for d in documents:
                    file_name = reverse_lookup_document_id.get(d, "Unknown")
                    print(file_name)

