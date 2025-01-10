import json
import math

import numpy as np


class NetMath:

    class TFIDFCalculator:
        def __init__(self, inverted_index, num_docs):
            self.tf_idf_matrix = {}
            self.inverted_index = inverted_index
            self.num_docs = num_docs
            self.weight_smoothing = 0.1
            self.max_weight_impact = 5.0

            # Weight multipliers for different fields
            self.weights = {
                'title': 3.0,  # Terms in title get 3x weight
                'genre': 2.0,  # Terms in genre get 2x weight
                'heading': 1.5,  # Terms in headings get 1.5x weight
                'content': 1.0  # Base weight for regular content
            }

        def calculate_term_freq(self, term, doc):
            # Retrieve the dictionary for the term and document
            term_data = self.inverted_index.get(term, {}).get(doc, {})

            # Retrieve the frequency data for the term in the document
            term_freq = term_data.get('frequency', 0)

            # Retrieve metadata for the document (title, genre, etc.)
            metadata = term_data.get('metadata', {})

            # Initialize the boosted term frequency with the regular frequency
            boosted_term_freq = term_freq

            # Check if the term exists in any of the metadata fields and apply a boost
            for field, values in metadata.items():
                # Check if the term is in the metadata field (case insensitive)
                if any(term.lower() == word.lower() for word in values):
                    boost = self.weights.get(field, 1.0)  # Apply the field's weight (default to 1.0)
                    boosted_term_freq += term_freq * boost  # Boost the term frequency

            print(f"[DEBUG] Boosted Term Frequency for '{term}' in '{doc}': {boosted_term_freq}")
            return boosted_term_freq

        def calculate_inverse_doc_freq(self, term):
            doc_freq = len(self.inverted_index.get(term, {}))
            idf = math.log((1 + self.num_docs) / (1 + doc_freq))
            # print(f"[DEBUG] Document Frequency for '{term}': {doc_freq}, IDF: {idf}")
            return idf

        def calculate_tf_idf(self, term, doc):
            """
            Calculate individual tf-idf values for each document
            :param term: Chosen term
            :param doc: Chosen document
            :return:
            """
            tf = self.calculate_term_freq(term, doc)
            idf = self.calculate_inverse_doc_freq(term)
            tf_idf = tf * idf
            # print(f"[DEBUG] TF-IDF for '{term}' in '{doc}': {tf_idf}")
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

        def save_tf_idf_to_file(self, file_name="json/tf_idf_matrix.json"):
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

            tf_idf = 0.0
            for term in query_tf:
                idf = self.calculate_inverse_doc_freq(term)
                tf_idf += query_tf[term] * idf
                query_tf_idf[term] = tf_idf

            return query_tf_idf




    class CosineSimilarity:
        def __init__(self):
            # Expanding queries Stores each query vector.
            self.query_vectors = []
            # No expanding queries right now.
            self.query_vector = None
            # Document vectors
            self.doc_vectors = None
            # Stores the size of our vocab
            self.vocab_size = 0
            # Before query expansion.
            self.query_tf_idf = None
            # Stores multiple query tf idfs for expanded query searching.
            self.query_tf_idfs = {}
            self.doc_tf_idf = None

        def calculate_cosine_similarity(self, doc_vector):
            """

            :param doc_vector: For specific document vectors.
            :return: Cosine Similarity between our query vector and document vector
            """
            dot_product = np.dot(self.query_vector, doc_vector)

            # normalize both
            query_norm = np.linalg.norm(self.query_vector)
            doc_norm = np.linalg.norm(self.doc_vectors[0][1])

            # prevent any division by 0

            if query_norm == 0 or doc_norm == 0:
                return 0.0

            return dot_product / (query_norm * doc_norm)

        def initialize_vectors(self, query_tf_idf, doc_tf_idf):
            """
            Convert TF-IDF scores for the query and documents into vectors and store them in class member variables.
            :param query_tf_idf: Dictionary of query TF-IDF values.
            :param vocab_size: Size of the vocabulary (number of terms).
            :param doc_tf_idf: Dictionary of document TF-IDF values.
            :return: query_vector, doc_vectors
            """
            # Store the input values as class member variables
            self.query_tf_idf = query_tf_idf
            self.doc_tf_idf = doc_tf_idf

            print(query_tf_idf)
            # essentially we do this to match the query with the document vocab this allows for our vector space

            vocab_terms = sorted(
                set(query_tf_idf.keys()).union(set(term for doc in doc_tf_idf.values() for term in doc.keys())))


            # Convert query TF-IDF to a numpy vector (one-dimensional array)
            self.query_vector = np.array([query_tf_idf.get(term, 0) for term in vocab_terms])

            # Convert all document TF-IDFs to numpy vectors (two-dimensional array)
            self.doc_vectors = []
            for doc_id, tf_idf_scores in doc_tf_idf.items():
                doc_vector = np.array([tf_idf_scores.get(term, 0) for term in vocab_terms])
                self.doc_vectors.append((doc_id, doc_vector))

            return self.query_vector, self.doc_vectors

        def initialize_vectors_weighted(self, query_tf_idfs, doc_tf_idf):
            """
            Convert TF-IDF scores for multiple queries and documents into vectors.

            :param query_tf_idfs: List of dictionaries containing TF-IDF values for each query
            :param doc_tf_idf: Dictionary of document TF-IDF values
            :return: query_vectors, doc_vectors
            """

            # Just debugging reasons i store in class member variables.
            self.query_tf_idfs = query_tf_idfs
            self.doc_tf_idf = doc_tf_idf

            # Get all unique terms from all queries and documents
            vocab_terms = set()
            for query_tf_idf in query_tf_idfs:
                vocab_terms.update(query_tf_idf.keys())
            for doc_tf_scores in doc_tf_idf.values():
                vocab_terms.update(doc_tf_scores.keys())

            # Convert to sorted list for consistent ordering
            vocab_terms = sorted(vocab_terms)

            # Convert each query TF-IDF to a vector
            self.query_vectors = []
            for query_tf_idf in query_tf_idfs:
                query_vector = np.array([query_tf_idf.get(term, 0) for term in vocab_terms])
                self.query_vectors.append(query_vector)

            # Convert document TF-IDFs to vectors
            self.doc_vectors = []
            for doc_id, tf_idf_scores in doc_tf_idf.items():
                doc_vector = np.array([tf_idf_scores.get(term, 0) for term in vocab_terms])
                self.doc_vectors.append((doc_id, doc_vector))

            return self.query_vectors, self.doc_vectors

        def calculate_query_weights(self, expanded_queries, original_weight = 0.4):
            """
            Perform our search with weighted queries, these weighted queries allow for our original query to have more
            importance over the others.
            :param expanded_queries: List of all queries (including our original query)
            :param original_weight: The original weight which is given to our original query
            :return: List of weights that will sum up to 1.0
            """
            number_of_e_queries = len(expanded_queries) - 1

            weights = [original_weight]

            # The remaining weight we need to distribute to the rest of the queries.
            remaining_weight = 1.0 - original_weight

            # Calculate weight for each expanded query
            expanded_query_weight = remaining_weight / number_of_e_queries

            # Add weights for expanded queries.
            weights.extend([expanded_query_weight] * number_of_e_queries)

            return weights

        def calculate_weighted_cosine_similarity(self, doc_vector, weights=None):
            """
            Calculate weighted cosine similarity between document and all query vectors.

            :param doc_vector: Document vector to compare against
            :param weights: List of weights for each query (defaults to equal weights)
            :return: Combined similarity score
            """
            if weights is None:
                weights = [1.0 / len(self.query_vectors)] * len(self.query_vectors)

            similarities = []
            for query_vector in self.query_vectors:
                dot_product = np.dot(query_vector, doc_vector)
                query_norm = np.linalg.norm(query_vector)
                doc_norm = np.linalg.norm(doc_vector)

                if query_norm == 0 or doc_norm == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(dot_product / (query_norm * doc_norm))

            # Combine similarities using weights
            final_similarity = sum(sim * weight for sim, weight in zip(similarities, weights))
            return final_similarity
    class BM25Calculator:
        def __init__(self, inverted_index, doc_lengths, k1=1.5, b=0.75):
            """
            Initialize BM25 calculator with necessary parameters

            Args:
                inverted_index: Dictionary mapping terms to documents and frequencies
                doc_lengths: Dictionary mapping document IDs to their lengths
                k1: Term frequency saturation parameter (default: 1.5)
                b: Length normalization parameter (default: 0.75)
            """
            self.inverted_index = inverted_index
            self.doc_lengths = doc_lengths
            self.k1 = k1
            self.b = b
            self.N = len(doc_lengths)  # Total number of documents
            self.avgdl = sum(doc_lengths.values()) / self.N  # Average document length

        def calculate_score(self, query_terms, doc_id):
            """
            Calculate BM25 score for a document given query terms

            Args:
                query_terms: List of query terms
                doc_id: Document identifier

            Returns:
                float: BM25 score
            """
            score = 0.0
            doc_length = self.doc_lengths[doc_id]

            for term in query_terms:
                if term not in self.inverted_index:
                    continue

                # Number of documents containing this term
                n_docs_term = len(self.inverted_index[term])

                # Get term frequency directly from nested structure
                tf = self.inverted_index[term].get(doc_id, {}).get('frequency', 0)

                # Calculate IDF
                idf = math.log((self.N - n_docs_term + 0.5) / (n_docs_term + 0.5) + 1.0)

                # Calculate normalized term frequency
                tf_normalized = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl))

                score += idf * tf_normalized

            return score

        def search(self, query, top_k=10):
            """
            Search documents using BM25 ranking

            Args:
                query: Search query string
                top_k: Number of top results to return

            Returns:
                List of (doc_id, score) tuples
            """
            query_terms = query.lower().split()
            scores = []

            # Calculate scores for all documents
            for doc_id in self.doc_lengths:
                score = self.calculate_score(query_terms, doc_id)
                scores.append((doc_id, score))

            # Sort by score in descending order
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]

# Barely got to implement this thought it would be fun...
    class SpellingCorrector:
        def __init__(self, vocabulary):
            self.vocabulary = vocabulary
            self.vocab_terms_dist = {}

        def lst_distance(self, s1: str, s2: str):
            s1 = ' ' + s1
            s2 = ' ' + s2

            m = np.zeros((len(s1), len(s2)), dtype=int)  # Ensure the matrix is initialized with integer type
            # Left-hand column and top row
            m[0, :] = np.arange(len(s2))
            m[:, 0] = np.arange(len(s1))

            for i in range(1, len(s1)):
                for j in range(1, len(s2)):
                    offset = 0 if s1[i] == s2[j] else 1
                    m[i][j] = min(m[i - 1][j] + 1, m[i][j - 1] + 1, m[i - 1][j - 1] + offset)

            return m[len(s1) - 1][len(s2) - 1]

        def generate_all_distance_vocab(self, term):
            if self.vocabulary is None:
                raise ValueError("Vocabulary is none... Please initialise the vocabulary first.")

            for vocab_term in self.vocabulary:
                self.vocab_terms_dist[vocab_term] = self.lst_distance(term, vocab_term)

            return self.vocab_terms_dist

        def write_all_edit_distance_vocab_json(self, output_file="vocab_distances.json"):
            # Convert numpy.int64 values to Python int
            converted_vocab_terms_dist = {k: int(v) for k, v in self.vocab_terms_dist.items()}

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(converted_vocab_terms_dist, f, ensure_ascii=False, indent=4)
                print(f"[INFO] Vocabulary distances saved to {output_file}")
            except Exception as e:
                print(f"[ERROR] Failed to write JSON: {e}")


        def correct_terms(self, term):
            if self.vocabulary is None:
                raise ValueError("Vocabulary is none... Please initialise the vocabulary first.")



            # This is a list of the terms with the distances switch between to get different info
            # However remember correct_terms is just a DEBUG method really.
            correct_terms = {}
            # This is just a list of the named terms without the distances
            correct_named_terms = []

            min_distance = float("inf")
            closest_term = term

            for vocab_term in self.vocabulary:
                distance = self.lst_distance(term, vocab_term)
                if distance == 1:
                    correct_terms[vocab_term] = distance
                    correct_named_terms.append(vocab_term)


            self.vocab_terms_dist = correct_terms
            correct_terms = sorted(correct_terms.items(), key=lambda x: x[1], reverse=False)
            return correct_named_terms

