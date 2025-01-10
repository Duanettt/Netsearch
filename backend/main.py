from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Union
import logging
from enum import Enum

from search_engine.search_engine_ref import TFIDFEngine, BM25Engine, NEREngine
from net_components.html_content_processor import read_folder_path, HTMLParser

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

use_query_expansion_var = True

# Pydantic models, these are used to identify which engines we can use and switch between.
class SearchEngineType(str, Enum):
    TFIDF = "tfidf"
    BM25 = "bm25" # Added option since I thought we could use it but not.
    NER = "ner"


class SearchResult(BaseModel):
    document_name: str
    similarity_score: float
    rank: int
    url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    expanded_queries: List[str]
    spell_checked_query: Optional[List[str]]
    entity_summary: Optional[Dict[str, int]]


class SearchQuery(BaseModel):
    query: str
    # [IMPORTANT] Change this when changing the Search Engine Type and then visit line 77 and comment out the other engines.
    engine_type: SearchEngineType = SearchEngineType.TFIDF
    max_results: Optional[int] = 10
    entity_boost: Optional[float] = 0.3
    k1: Optional[float] = 1.5
    b: Optional[float] = 0.75
    use_query_expansion: Optional[bool] = use_query_expansion_var


class SearchApp:
    def __init__(self):
        self.search_engines = None
        self.app = FastAPI()
        self.setup_cors()
        self.initialize_search_engines()
        self.setup_routes()

    def setup_cors(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def initialize_search_engines(self):
        # Initialize HTML parser
        folder_path = './videogames/'
        file_path_list = read_folder_path(folder_path)
        self.html_parser = HTMLParser(file_path_list)
        self.html_parser.set_use_json(True) # This enables whether you load json data from directories
        self.html_parser.set_use_stemming(False) # This enables whether you use stemming
        self.html_parser.set_use_lemmatizer(False) # This enables whether you use lemmatizer.
        self.html_parser.parse_and_process_html()
        # self.html_parser.save_to_json()

        # Initialize different search engines
        self.search_engines = {
            SearchEngineType.TFIDF: TFIDFEngine(self.html_parser),
            # SearchEngineType.BM25: BM25Engine(self.html_parser),
            # SearchEngineType.NER: NEREngine(self.html_parser)
        }

        # Build indices for each engine
        for engine in self.search_engines.values():
            engine.set_json(False) # This enables whether you want to use json with the inverted index.
            engine.build_inverted_index()
            if isinstance(engine, TFIDFEngine):
                engine.debug_tf_idf()

# Routes:
    def setup_routes(self):
        @self.app.get("/")
        async def root():
            return {"message": "Search Engine API is running"}

        @self.app.post("/search", response_model=SearchResponse)
        async def search(search_query: SearchQuery):
            try:
                logger.debug(f"Received search query: {search_query}")
                engine = self.search_engines[search_query.engine_type]

                # Get spell-checked query if available
                spell_checked = []
                if hasattr(engine, 'spelling_corrector'):
                    spell_checked = engine.spelling_corrector.correct_terms(search_query.query)
                    logger.debug(f"Spell checked query: {spell_checked}")


                if isinstance(engine, TFIDFEngine):
                    engine.set_query_expansion(use_query_expansion_var)

                # Extract entities if using NER engine
                entity_summary = {}
                if isinstance(engine, NEREngine):
                    query_entities = engine.extract_entities(search_query.query)
                    logger.debug(f"Extracted entities: {query_entities}")
                    for entity_text, entity_type in query_entities:
                        entity_summary[entity_type] = entity_summary.get(entity_type, 0) + 1

                # Update BM25 parameters if applicable
                if isinstance(engine, BM25Engine) and search_query.k1 and search_query.b:
                    engine.bm25_calculator.k1 = search_query.k1
                    engine.bm25_calculator.b = search_query.b

                # Perform search
                similarities = engine.search(search_query.query)

                # Prepare results
                results = self._prepare_results(
                    similarities[:search_query.max_results],
                    engine
                )


                logger.debug(f"Results prepared: {len(results)} results")

                return SearchResponse(
                    results=results,
                    expanded_queries=[],
                    spell_checked_query=spell_checked,
                    entity_summary=entity_summary
                )

            except Exception as e:
                logger.exception("Search error occurred")
                return JSONResponse(
                    status_code=500,
                    content={"detail": str(e), "type": str(type(e))}
                )

        @self.app.get("/stats")
        async def get_stats():
            stats = {}
            for engine_type, engine in self.search_engines.items():
                stats[engine_type] = {
                    "total_documents": len(engine.docIDs),
                    "vocabulary_size": len(engine.vocab),
                    "unique_terms": len(engine.find_unique_terms())
                }
                if isinstance(engine, NEREngine):
                    stats[engine_type].update({
                        "entity_types": list(engine.entity_index.keys()),
                        "total_entities": sum(len(entities) for entities in engine.entity_index.values())
                    })
            return stats

        @self.app.get("/entities/{entity_type}")
        async def get_entities_by_type(entity_type: str):
            ner_engine = self.search_engines[SearchEngineType.NER]
            if entity_type not in ner_engine.entity_index:
                raise HTTPException(
                    status_code=404,
                    detail=f"Entity type '{entity_type}' not found"
                )
            return {
                "entity_type": entity_type,
                "entities": list(ner_engine.entity_index[entity_type].keys())
            }

        @self.app.get("/bm25-params")
        async def get_bm25_params():
            bm25_engine = self.search_engines[SearchEngineType.BM25]
            return {
                "k1": bm25_engine.bm25_calculator.k1,
                "b": bm25_engine.bm25_calculator.b,
                "average_doc_length": bm25_engine.bm25_calculator.avgdl
            }

    def _prepare_results(self, similarities, engine) -> List[SearchResult]:
        results = []
        reverse_lookup_document_id = {v: k for k, v in engine.docIDs.items()}

        for rank, (doc_id, similarity) in enumerate(similarities, 1):
            doc_id_lookup = engine.docIDs[doc_id]
            doc_name = reverse_lookup_document_id.get(doc_id_lookup, "Unknown")

            # Get URL from categorized data
            url = None
            for title, docs in self.html_parser.categorized_data.items():
                for doc in docs:
                    if doc['file'] == doc_name:
                        url = doc['url']
                        break
                if url:
                    break

            results.append(SearchResult(
                document_name=doc_name,
                similarity_score=float(similarity),
                rank=rank,
                url=url or f"file://{doc_name}"
            ))

        return results

    def run(self, host="localhost", port=5000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


if __name__ == "__main__":
    search_app = SearchApp()
    search_app.run()