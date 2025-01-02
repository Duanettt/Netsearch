import json
import math
import os
from collections import defaultdict

from nltk import ngrams

import html_content_processor
import net_util
from html_content_processor import HTMLParser
from html_content_processor import WebScraper
from html_content_processor import read_folder_path

from net_math import NetMath
from net_util import expand_query
from flask import Flask, request, jsonify

app = Flask(__name__)


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

    def build_inverted_index(self, use_json_inverted_index = False):
        categorized_data = self.HTMLParser.categorized_data
        # Basically allows us to just load the inverted_index we built already see if we can shave off time.
        if use_json_inverted_index:
            self.load_inverted_index()
            self.load_docIDs()
        else:
            for title, files in categorized_data.items():
                for file in files:
                    file_name = file['file']
                    tokens = file['tokens']
                    url = file['url']
                    self.update_inverted_index(file_name, tokens)

        print(len(self.docIDs))
        self.write_inverted_index_to_file()
        self.write_docIDs()
        self.tfidf_calculator = NetMath.TFIDFCalculator(self.inverted_index, len(self.docIDs))
        self.cosine_sim_calculator = NetMath.CosineSimilarity()
        self.write_vocab()

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
            token = token.lower()
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

    def write_inverted_index_to_file(self): # Ensure postings are finalized before saving
        self.finalize_postings()
        with open("inverted_index.json", 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=4)

    def write_docIDs(self): # Ensure postings are finalized before saving
        with open("doc_ids.json", 'w', encoding='utf-8') as f:
            json.dump(self.docIDs, f, ensure_ascii=False, indent=4)

    def write_vocab(self, output_file="vocab.json"):
        """
        Writes the vocabulary to a specified JSON file.

        :param output_file: Name of the file where the vocabulary will be saved.
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False, indent=4)
            print(f"Vocabulary successfully written to {output_file}.")
        except Exception as e:
            print(f"An error occurred while writing the vocabulary: {e}")

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

            # expanded_queries = net_util.expand_query(query)

            # print(expanded_queries)

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
            counter = 0
            for doc_id, similarity in similarities:
                if counter >= 10:
                    break
                doc_id = self.docIDs[doc_id]
                doc_name = reverse_lookup_document_id.get(doc_id, "Unknown")
                print(f"Document: {doc_name}, Cosine Similarity: {similarity}")
                counter += 1

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

    def load_inverted_index(self, file_path = "inverted_index.json"):
        try:
            # Load inverted index
            with open(file_path, 'r', encoding='utf-8') as f:
                self.inverted_index = json.load(f)
            print(f"[INFO] Loaded inverted index with {len(self.inverted_index)} terms")
            return True



        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error loading index: {e}")
            return False

    def load_docIDs(self, file_path = "doc_ids.json"):
        try:
            # Load inverted index
            with open(file_path, 'r', encoding='utf-8') as f:
                self.docIDs = json.load(f)
            print(f"[INFO] Loaded inverted index with {len(self.inverted_index)} terms")
            return True

        except FileNotFoundError as e:
            print(f"[ERROR] File not found: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error loading index: {e}")
            return False

from spacy import load
from typing import Dict, List, Set, Tuple


class NEREnhancedSearch(SearchEngine):
    def __init__(self, html_parser):
        super().__init__(html_parser)
        # load spacy model
        self.nlp = load("en_core_web_sm")
        # looks crazy confusing but found it online.
        # we store a dictionary of our entity type and the term, we then have the set of documents it belongs to.
        # self.entity_index: [[[]]] = {}  this is what it looks like so thought it would be more clear to put the types.
        self.entity_index: Dict[str, Dict[str, Set[str]]] = {}

    def extract_entities(self, text):
        """Extract entities and their types from text using SpaCy."""
        # we process our text
        doc = self.nlp(text)
        # we then return turn a list of our
        return [(ent.text.lower(), ent.label_) for ent in doc.ents]

    def update_inverted_index(self, file_name, tokens):
        """Override parent method to include entity extraction."""
        # We update our inverted index which should return better tokens.
        super().update_inverted_index(file_name, tokens)

        # Put all of our tokens into a text string
        text = " ".join(tokens)
        entities = self.extract_entities(text)

        # Now we really update our inverted_index to take into account the NER
        # so essentially
        for entity_text, entity_type in entities:
            if entity_type not in self.entity_index:
                self.entity_index[entity_type] = {}

            if entity_text not in self.entity_index[entity_type]:
                self.entity_index[entity_type][entity_text] = set()

            self.entity_index[entity_type][entity_text].add(file_name)

    def search(self, query: str) -> List[Tuple[str, float]]:
        """Enhanced search considering both regular tokens and named entities."""
        # Extract entities from the query
        query_entities = self.extract_entities(query)

        expanded_queries = net_util.expand_query(query)
        expanded_queries = expand_query(query).split(',')
        expanded_queries = [q.strip() for q in expanded_queries]
        expanded_queries.insert(0, query)  # Add original query

        print(f"[INFO] Query expanded to: {expanded_queries}")
        # Get regular TF-IDF results
        # iterate through all the queries and generate final scores.
        query_tf_idfs = []
        for q in expanded_queries:
            query_tf_idf = self.tfidf_calculator.calculate_query_tf_idf(q)
            query_tf_idfs.append(query_tf_idf)

            query_vectors, doc_vectors = self.cosine_sim_calculator.initialize_vectors_weighted(
            query_tf_idfs,
            self.tfidf_calculator.tf_idf_matrix
            )


        # weights so we can highlight the importance of the first query.
        weights = self.cosine_sim_calculator.calculate_query_weights(expanded_queries, original_weight=0.4)
        # Calculate base similarities
        base_similarities = []
        for doc_id, doc_vector in doc_vectors:
            cosine_sim = self.cosine_sim_calculator.calculate_weighted_cosine_similarity(doc_vector, weights)
            base_similarities.append((doc_id, cosine_sim))

        # Boost scores for documents containing matching entities
        boosted_similarities = []
        entity_boost = 0.3  # Configurable boost factor


        # NER testing basically but we boost if its a named entity
        for doc_id, base_score in base_similarities:
            boost = 0
            # Nested for loops might modify.
            for entity_text, entity_type in query_entities:
                if (entity_type in self.entity_index and
                        entity_text in self.entity_index[entity_type] and
                        doc_id in self.entity_index[entity_type][entity_text]):
                    boost += entity_boost

            final_score = min(1.0, base_score + boost)  # Cap at 1.0
            boosted_similarities.append((doc_id, final_score))

        # Sort by final score
        boosted_similarities.sort(key=lambda x: x[1], reverse=True)
        return boosted_similarities


    # Needed for testing comparison.
    def no_expansion_search(self, query: str) -> List[Tuple[str, float]]:
        """Enhanced search considering both regular tokens and named entities."""
        # Extract entities from the query
        query_entities = self.extract_entities(query)

        # Get regular TF-IDF results
        query_tf_idf = self.tfidf_calculator.calculate_query_tf_idf(query)
        query_vector, doc_vectors = self.cosine_sim_calculator.initialize_vectors(
            query_tf_idf,
            self.tfidf_calculator.tf_idf_matrix
        )

        # Calculate base similarities
        base_similarities = []
        for doc_id, doc_vector in doc_vectors:
            cosine_sim = self.cosine_sim_calculator.calculate_cosine_similarity(doc_vector)
            base_similarities.append((doc_id, cosine_sim))

        # Boost scores for documents containing matching entities
        boosted_similarities = []
        entity_boost = 0.3  # Configurable boost factor

        for doc_id, base_score in base_similarities:
            boost = 0

            # Check for entity matches
            for entity_text, entity_type in query_entities:
                if (entity_type in self.entity_index and
                        entity_text in self.entity_index[entity_type] and
                        doc_id in self.entity_index[entity_type][entity_text]):
                    boost += entity_boost

            final_score = min(1.0, base_score + boost)  # Cap at 1.0
            boosted_similarities.append((doc_id, final_score))

        # Sort by final score
        boosted_similarities.sort(key=lambda x: x[1], reverse=True)
        return boosted_similarities

    def write_entity_index_to_file(self, filename: str = "entity_index.json"):
        """Save the entity index to a JSON file."""
        # Convert sets to lists for JSON serialization
        serializable_index = {}
        for entity_type, entities in self.entity_index.items():
            serializable_index[entity_type] = {
                entity: list(docs)
                for entity, docs in entities.items()
            }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=4)
