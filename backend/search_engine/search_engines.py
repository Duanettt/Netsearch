from typing import List, Tuple, Dict, Set
import json
from spacy import load
import net_util
from net_components.net_math import NetMath


class BaseSearchEngine:
    def __init__(self, html_parser):
        self.HTMLParser = html_parser
        self.inverted_index = {}
        self.vocab = {}
        self.docIDs = {}
        self.postings = {}
        self.doc_lengths = {}
        self.vocab_counter = 0
        self.doc_counter = 0
        self.use_json = False


        self.term_locations = {}

    def set_json(self, enabled : bool):
        use_json = enabled

    def build_inverted_index(self):
        if self.use_json:
            self.load_inverted_index()
            self.load_docIDs()
        else:
            categorized_data = self.HTMLParser.categorized_data
            for title, files in categorized_data.items():
                for file in files:
                    self.update_inverted_index(file['file'], file['tokens'])

        print(f"Built index with {len(self.docIDs)} documents")
        self.write_inverted_index_to_file()
        self.write_docIDs()
        self.write_vocab()

    def update_inverted_index(self, file_name: str, tokens: List[str]):
        self.doc_lengths[file_name] = len(tokens)

        if file_name not in self.docIDs:
            self.docIDs[file_name] = self.doc_counter
            docID = self.doc_counter
            self.doc_counter += 1
        else:
            docID = self.docIDs[file_name]

        doc_metadata = self._get_document_metadata(file_name)

        # Add metadata to the inverted index for this document
        for token in tokens:
            token = token.lower()

            # Add term to the vocab if it's not already present
            if token not in self.vocab:
                self.vocab[token] = self.vocab_counter
                vocabID = self.vocab_counter
                self.vocab_counter += 1
            else:
                vocabID = self.vocab[token]

            # Add posting to the inverted index (with document ID)
            if vocabID not in self.postings:
                self.postings[vocabID] = {docID}
            else:
                self.postings[vocabID].add(docID)

            # Track term locations and metadata (title, genre, etc.)
            if token not in self.term_locations:
                self.term_locations[token] = {}
            if docID not in self.term_locations[token]:
                self.term_locations[token][docID] = {
                    'title': 0, 'genre': 0, 'heading': 0,
                }

            # Update the inverted index with term frequency and metadata for the document
            if token not in self.inverted_index:
                self.inverted_index[token] = {}

            if file_name not in self.inverted_index[token]:
                self.inverted_index[token][file_name] = {
                    'frequency': 1,
                    'metadata': doc_metadata  # Add metadata for this document
                }
            else:
                self.inverted_index[token][file_name]['frequency'] += 1

    def _get_document_metadata(self, file_name: str) -> Dict:
        for title, files in self.HTMLParser.categorized_data.items():
            for file in files:
                if file['file'] == file_name:
                    return {
                        'title': title.lower().split(),
                        'genre': file['game_info'].get('Genre', '').lower().split(),
                        'publisher': file['game_info'].get('Publisher', '').lower().split(),
                    }
        return {'title': [], 'genre': [], 'publisher': []}



    def search(self, query: str) -> List[Tuple[str, float]]:
        raise NotImplementedError("Subclasses must implement search method")

    # File I/O operations
    def write_inverted_index_to_file(self):
        self.finalize_postings()
        with open("json/inverted_index.json", 'w', encoding='utf-8') as f:
            json.dump(self.inverted_index, f, ensure_ascii=False, indent=4)

    def write_docIDs(self):
        with open("json/doc_ids.json", 'w', encoding='utf-8') as f:
            json.dump(self.docIDs, f, ensure_ascii=False, indent=4)

    def write_vocab(self, output_file="json/vocab.json"):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    def load_inverted_index(self, file_path="json/inverted_index.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.inverted_index = json.load(f)
            print(f"[INFO] Loaded inverted index with {len(self.inverted_index)} terms")
            return True
        except Exception as e:
            print(f"[ERROR] Error loading index: {e}")
            return False

    def load_docIDs(self, file_path="json/doc_ids.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.docIDs = json.load(f)
            return True
        except Exception as e:
            print(f"[ERROR] Error loading docIDs: {e}")
            return False

    def finalize_postings(self):
        for vocabID in self.postings:
            self.postings[vocabID] = sorted(self.postings[vocabID])


class TFIDFEngine(BaseSearchEngine):
    def __init__(self, html_parser):
        super().__init__(html_parser)
        print(f"[INFO] Initializing TF-IDF Engine")
        self.tfidf_calculator = None
        self.cosine_sim_calculator = None
        self.spelling_corrector = None

    def build_inverted_index(self):
        super().build_inverted_index()
        print(f"[INFO] Updating Building Inverted Index for TFIDFEngine Engine")
        self.tfidf_calculator = NetMath.TFIDFCalculator(self.inverted_index, len(self.docIDs))
        self.cosine_sim_calculator = NetMath.CosineSimilarity()
        self.spelling_corrector = NetMath.SpellingCorrector(self.vocab)
        self.use_query_expansion = True

    def set_query_expansion(self, enabled : bool):
        self.use_query_expansion = enabled

    def search(self, query: str) -> List[Tuple[str, float]]:
        if self.use_query_expansion:
            # Handle expanded queries with weighted calculations
            expanded_queries = net_util.expand_query(query).split(',')
            expanded_queries = [q.strip() for q in expanded_queries]
            expanded_queries.insert(0, query)
            print(f"[INFO] Query expanded to: {expanded_queries}")

            query_tf_idfs = [
                self.tfidf_calculator.calculate_query_tf_idf(q)
                for q in expanded_queries
            ]

            query_vectors, doc_vectors = self.cosine_sim_calculator.initialize_vectors_weighted(
                query_tf_idfs,
                self.tfidf_calculator.tf_idf_matrix
            )

            weights = self.cosine_sim_calculator.calculate_query_weights(
                expanded_queries,
                original_weight=0.4
            )

            similarities = [
                (doc_id, self.cosine_sim_calculator.calculate_weighted_cosine_similarity(doc_vector, weights))
                for doc_id, doc_vector in doc_vectors
            ]
        else:
            # Handle single query with standard cosine similarity
            print("[INFO] Query expansion disabled")
            query_tf_idf = self.tfidf_calculator.calculate_query_tf_idf(query)

            # Use regular vector initialization for single query
            query_vector, doc_vectors = self.cosine_sim_calculator.initialize_vectors(
                query_tf_idf,
                self.tfidf_calculator.tf_idf_matrix
            )

            # Use standard cosine similarity calculation
            similarities = [
                (doc_id, self.cosine_sim_calculator.calculate_cosine_similarity(doc_vector))
                for doc_id, doc_vector in doc_vectors
            ]

        return sorted(similarities, key=lambda x: x[1], reverse=True)


    def debug_tf_idf(self):
        self.tfidf_calculator.calculate_all_tf_idf()
        self.tfidf_calculator.save_tf_idf_to_file()


class BM25Engine(BaseSearchEngine):
    def __init__(self, html_parser):
        super().__init__(html_parser)
        print(f"[INFO] Initializing BM25 Engine")
        self.bm25_calculator = None

    def initialize_bm25(self):
        """Initialize BM25 calculator after building inverted index"""
        self.bm25_calculator = NetMath.BM25Calculator(
            self.inverted_index,
            self.docIDs,
            self.doc_lengths
        )

    def build_inverted_index(self):
        super().build_inverted_index()
        print(f"[INFO] Updating Building Inverted Index for BM25 Engine")

        self.bm25_calculator = NetMath.BM25Calculator(
            self.inverted_index,
            self.docIDs,
            self.doc_lengths
        )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.bm25_calculator:
            self.initialize_bm25()
        return self.bm25_calculator.search(query, top_k)


class NEREngine(TFIDFEngine):
    def __init__(self, html_parser):
        super().__init__(html_parser)
        self.nlp = load("en_core_web_sm")
        self.entity_index: Dict[str, Dict[str, Set[str]]] = {}

    def extract_entities(self, text: str):
        doc = self.nlp(text)
        return [(ent.text.lower(), ent.label_) for ent in doc.ents]

    def update_inverted_index(self, file_name: str, tokens: List[str]):
        super().update_inverted_index(file_name, tokens)

        text = " ".join(tokens)
        entities = self.extract_entities(text)

        # Basically identical to the inverted index
        for entity_text, entity_type in entities:
            if entity_type not in self.entity_index:
                self.entity_index[entity_type] = {}

            if entity_text not in self.entity_index[entity_type]:
                self.entity_index[entity_type][entity_text] = set()

            self.entity_index[entity_type][entity_text].add(file_name)

        self.write_entity_index_to_file()

    def search(self, query: str) -> List[Tuple[str, float]]:
        # we use more nlp to extract entities using our query.
        query_entities = self.extract_entities(query)
        # we use our TFIDF engine search to do the calculations
        base_similarities = super().search(query)
        # Entity boost is defined
        entity_boost = 0.3
        boosted_similarities = []
        # we check if any of our query entities is within any of the types of the texts. We then apply a boost and sum it
        # We then add this final boost.
        for doc_id, base_score in base_similarities:
            boost = sum(
                entity_boost
                for entity_text, entity_type in query_entities
                if (entity_type in self.entity_index and
                    entity_text in self.entity_index[entity_type] and
                    doc_id in self.entity_index[entity_type][entity_text])
            )

            final_score = min(1.0, base_score + boost)
            boosted_similarities.append((doc_id, final_score))

        return sorted(boosted_similarities, key=lambda x: x[1], reverse=True)

    def write_entity_index_to_file(self, filename: str = "json/entity_index.json"):
        serializable_index = {
            entity_type: {
                entity: list(docs)
                for entity, docs in entities.items()
            }
            for entity_type, entities in self.entity_index.items()
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=4)