from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import json
import logging

from search_engine import NEREnhancedSearch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from html_content_processor import HTMLParser, read_folder_path

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize NER-enhanced search engine
folder_path = './videogames/'
file_path_list = read_folder_path(folder_path)
user_parser = HTMLParser(file_path_list)
user_parser.parse_and_process_html(True)

search_engine = NEREnhancedSearch(user_parser)  # Use NEREnhancedSearch instead of SearchEngine
search_engine.build_inverted_index(True)
search_engine.debug_tf_idf()
search_engine.write_entity_index_to_file()


# Enhanced response models

# Entity Match used for Debugging NER
# class EntityMatch(BaseModel):
#     entity_text: str
#     entity_type: str
#     documents: List[str]


class SearchResult(BaseModel):
    document_name: str
    similarity_score: float
    rank: int
    url: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    expanded_queries: List[str]
    spell_checked_query: List[str]
    entity_summary: Dict[str, int]  # Summary of entity types found in query


class SearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = 10
    entity_boost: Optional[float] = 0.3  # Allow configurable entity boost


@app.get("/")
async def root():
    return {"message": "NER-Enhanced Search Engine API is running"}


@app.post("/search", response_model=SearchResponse)
async def search(search_query: SearchQuery):
    try:
        logger.debug(f"Received search query: {search_query}")

        # Get spell-checked query
        spell_checked = search_engine.spelling_corrector.correct_terms(search_query.query)
        logger.debug(f"Spell checked query: {spell_checked}")

        # Extract entities from query
        query_entities = search_engine.extract_entities(search_query.query)
        logger.debug(f"Extracted entities: {query_entities}")

        # Get search results with entity boosting
        similarities = search_engine.search(search_query.query)

        # Prepare results with entity information
        results = []
        reverse_lookup_document_id = {v: k for k, v in search_engine.docIDs.items()}

        for rank, (doc_id, similarity) in enumerate(similarities[:search_query.max_results], 1):
            doc_id_lookup = search_engine.docIDs[doc_id]
            doc_name = reverse_lookup_document_id.get(doc_id_lookup, "Unknown")

            # Get URL from categorized data
            url = None
            for title, docs in user_parser.categorized_data.items():
                for doc in docs:
                    if doc['file'] == doc_name:
                        url = doc['url']
                        break
                if url:
                    break

            # # Get matched entities for this document
            # matched_entities = []
            # for entity_text, entity_type in query_entities:
            #     if (entity_type in search_engine.entity_index and
            #             entity_text in search_engine.entity_index[entity_type] and
            #             doc_id in search_engine.entity_index[entity_type][entity_text]):
            #         matched_entities.append(EntityMatch(
            #             entity_text=entity_text,
            #             entity_type=entity_type,
            #             documents=[doc_name]
            #         ))

            results.append(SearchResult(
                document_name=doc_name,
                similarity_score=float(similarity),
                rank=rank,
                # matched_entities=matched_entities,
                url=url or f"file://{doc_name}"  # Fallback if URL not found
            ))

        # Create entity summary
        entity_summary = {}
        for entity_text, entity_type in query_entities:
            entity_summary[entity_type] = entity_summary.get(entity_type, 0) + 1

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
@app.get("/stats")
async def get_stats():
    return {
        "total_documents": len(search_engine.docIDs),
        "vocabulary_size": len(search_engine.vocab),
        "unique_terms": search_engine.find_unique_terms(),
        "entity_types": list(search_engine.entity_index.keys()),
        "total_entities": sum(len(entities) for entities in search_engine.entity_index.values())
    }


@app.get("/entities/{entity_type}")
async def get_entities_by_type(entity_type: str):
    if entity_type not in search_engine.entity_index:
        raise HTTPException(status_code=404, detail=f"Entity type '{entity_type}' not found")

    return {
        "entity_type": entity_type,
        "entities": list(search_engine.entity_index[entity_type].keys())
    }





if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=5000)