import json
import os

import html_content_processor
from data_formatter import TokensDataStorage
from html_content_processor import HTMLParser
from html_content_processor import WebScraper
from html_content_processor import read_folder_path

import data_formatter
from net_tf_Idf_calculator import NetMath


class SearchEngine:
    def __init__(self, html_parser):
        self.tf_idf_matrix = {}
        self.HTMLParser = html_parser
        self.tokenized_data = []
        self.inverted_index = {}
        self.vocab = {}
        self.docIDs = {}
        self.postings = {}

        # TFIDF calculations (Change so they all initialise our tf_idf calculator and stuff)
        self.tfidf_calculator = None
        self.cosine_sim_calculator = None
        self.spelling_corrector = NetMath.SpellingCorrector(self.vocab)

        self.vocab_counter = 0   # Tracks unique ID for each vocab term
        self.doc_counter = 0     # Tracks unique ID for each document

    def build_inverted_index(self):
        categorized_data = self.HTMLParser.categorized_data
        for title, files in categorized_data.items():
            for file in files:
                file_name = file['file']
                tokens = file['tokens']
                self.update_inverted_index(file_name, tokens)

        print(len(self.docIDs))
        self.tfidf_calculator = NetMath.TFIDFCalculator(self.inverted_index, len(self.docIDs))
        self.cosine_sim_calculator = NetMath.CosineSimilarity()

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
            query_tf_idf = self.tfidf_calculator.calculate_query_tf_idf(query)
            print(query_tf_idf)
            if query == "quit":
                break
            if query in self.vocab:
                vocabID = self.vocab[query]
                documents = self.postings[vocabID]
                for d in documents:
                    file_name = reverse_lookup_document_id.get(d, "Unknown")
                    print(file_name)

    def user_prompt_tfidf(self):
        reverse_lookup_document_id = {v: k for k, v in self.docIDs.items()}

        while True:
            query = input("What would you like to search for? ")

            self.spelling_corrector.correct_terms(query)
            self.spelling_corrector.write_all_edit_distance_vocab_json()

            # Step 1: Get the query's TF-IDF values
            query_tf_idf = self.tfidf_calculator.calculate_query_tf_idf(query)

            # Step 2: Initialize the query and document vectors
            query_vector, doc_vectors = self.cosine_sim_calculator.initialize_vectors(query_tf_idf,self.tfidf_calculator.tf_idf_matrix)

            # Step 3: Calculate cosine similarities for the query with each document
            similarities = []
            for doc_id, doc_vector in doc_vectors:
                cosine_sim = self.cosine_sim_calculator.calculate_cosine_similarity(doc_vector)
                similarities.append((doc_id, cosine_sim))

            # Step 4: Sort the documents by cosine similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Step 5: Print the sorted results (documents and their similarity scores)
            for doc_id, similarity in similarities:
                doc_id = self.docIDs[doc_id]
                doc_name = reverse_lookup_document_id.get(doc_id, "Unknown")
                print(f"Document: {doc_name}, Cosine Similarity: {similarity}")

            if query == "quit":
                break

    # Utility functions
    def find_unique_terms(self):
        unique_terms = []
        for term, docs in self.inverted_index.items():
            if len(docs) == 1:  # Only one document contains this term
                unique_terms.append(term)
        return unique_terms

    def debug_tf_idf(self):
        self.tfidf_calculator.calculate_all_tf_idf()
        self.tfidf_calculator.save_tf_idf_to_file()