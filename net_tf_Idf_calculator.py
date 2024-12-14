import json
import math

import numpy as np


class NetMath:

    class TFIDFCalculator:
        def __init__(self, inverted_index, num_docs):
            self.tf_idf_matrix = {}
            self.inverted_index = inverted_index
            self.num_docs = num_docs

        def calculate_term_freq(self, term, doc):
            term_freq = self.inverted_index.get(term, {}).get(doc, 0)
            print(f"[DEBUG] Term Frequency for '{term}' in '{doc}': {term_freq}")
            return term_freq

        def calculate_inverse_doc_freq(self, term):
            doc_freq = len(self.inverted_index.get(term, {}))
            idf = math.log((1 + self.num_docs) / (1 + doc_freq))
            print(f"[DEBUG] Document Frequency for '{term}': {doc_freq}, IDF: {idf}")
            return idf

        def calculate_tf_idf(self, term, doc):
            tf = self.calculate_term_freq(term, doc)
            idf = self.calculate_inverse_doc_freq(term)
            tf_idf = tf * idf
            print(f"[DEBUG] TF-IDF for '{term}' in '{doc}': {tf_idf}")
            return tf_idf

        def calculate_all_tf_idf(self):
            """
            Calculate TF-IDF values for all terms in all documents and store them in a matrix.
            """
            if not self.inverted_index:
                print("[ERROR] TFIDFCalculator not initialized. Build the inverted index first.")
                return

            print("[INFO] Calculating TF-IDF for all terms in all documents...")
            for term, docs in self.inverted_index.items():
                for doc in docs:
                    tf_idf_value = self.calculate_tf_idf(term, doc)
                    if doc not in self.tf_idf_matrix:
                        self.tf_idf_matrix[doc] = {}
                    self.tf_idf_matrix[doc][term] = tf_idf_value

            print(f"[INFO] Completed TF-IDF calculations for {len(self.tf_idf_matrix)} documents.")

        def save_tf_idf_to_file(self, file_name="tf_idf_matrix.json"):
            """
            Save the TF-IDF matrix to a file for inspection or further use.
            """
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(self.tf_idf_matrix, f, ensure_ascii=False, indent=4)
            print(f"[INFO] TF-IDF matrix saved to {file_name}")

        def calculate_query_tf_idf(self, query):
            # we tokenise our query
            query_tokens = query.lower().split()

            # calculate the term frequency within our query.
            total_terms = len(query_tokens)
            term_counts = {}

            # we then count the number of terms.
            for term in query_tokens:
                term_counts[term] = term_counts.get(term, 0) + 1

            query_tf = {term: count / total_terms for term, count in term_counts.items()}

            query_tf_idf = {}

            for term in query_tf:
                idf = self.calculate_inverse_doc_freq(term)
                tf_idf = query_tf[term] * idf
                query_tf_idf[term] = tf_idf

            print(query_tf_idf)

            return query_tf_idf

    class CosineSimilarity:
        def __init__(self):
            self.query_vector = None
            self.doc_vectors = None
            self.vocab_size = 0
            self.query_tf_idf = None
            self.doc_tf_idf = None

        def calculate_cosine_similarity(self, query_tf_idf, doc_tf_idf):
            dot_product = np.dot(self.query_vector, self.doc_vectors[0][1])

            # normalize both
            query_norm = np.linalg.norm(self.query_vector)
            doc_norm = np.linalg.norm(self.doc_vectors[0][1])

            # prevent any division by 0

            if query_norm == 0 or doc_norm == 0:
                return 0.0

            return dot_product / (query_norm * doc_norm)

        def initialize_vectors(self, query_tf_idf, vocab_size, doc_tf_idf):
            """
            Convert TF-IDF scores for the query and documents into vectors and store them in class member variables.
            :param query_tf_idf: Dictionary of query TF-IDF values.
            :param vocab_size: Size of the vocabulary (number of terms).
            :param doc_tf_idf: Dictionary of document TF-IDF values.
            :return: query_vector, doc_vectors
            """
            # Store the input values as class member variables
            self.query_tf_idf = query_tf_idf
            self.vocab_size = vocab_size
            self.doc_tf_idf = doc_tf_idf

            # Convert query TF-IDF to a numpy vector (one-dimensional array)
            self.query_vector = np.array([query_tf_idf.get(term, 0) for term in range(vocab_size)])

            # Convert all document TF-IDFs to numpy vectors (two-dimensional array)
            self.doc_vectors = []
            for doc_id, tf_idf_scores in doc_tf_idf:
                doc_vector = np.array([tf_idf_scores.get(term, 0) for term in range(vocab_size)])
                self.doc_vectors.append((doc_id, doc_vector))

            return self.query_vector, self.doc_vectors