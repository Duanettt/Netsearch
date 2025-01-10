# Netsearch

## Welcome to Netsearch! 🎮🚀

Netsearch is an innovative Game Search Engine built with **React Three Fiber** and **Python**. Whether you're a casual gamer or a hardcore enthusiast, Netsearch aims to provide an immersive, efficient, and robust search experience for discovering your next favorite game.

## 🚀 Features

* **Interactive 3D Interface**: Explore games in a dynamic 3D environment powered by React Three Fiber.
* **Advanced Search Capabilities**: Leverage powerful algorithms like **TF-IDF** and **Cosine Similarity** for precise, relevant search results.
* **Named Entity Recognition (NER)**: Enhance search results by identifying and prioritizing game-related entities such as genres, developers, platforms, and more.
* **Efficient Query Expansion**: Use **LLM-based query expansion** to refine and broaden your search results for more accuracy (Unavailable currently unless you have .env)

## 🌟 Technologies Used

* **React Three Fiber**: A powerful React renderer for Three.js that powers our interactive 3D search interface.
* **Python (FastAPI, SpaCy)**: The backend is powered by Python for processing and serving search results, with **SpaCy** for NLP tasks like tokenization, stemming, and entity extraction.
* **TF-IDF & Cosine Similarity**: Core algorithms that help rank game titles based on your search queries.
* **Named Entity Recognition**: A modern NLP technique for boosting entity relevance in search results.

## 💡 How It Works

1. **Search**: Enter a query (e.g., "Action RPGs" or "Strategy games for PC").
2. **Process**: The system processes the query using TF-IDF for term weighting, Cosine Similarity for ranking, and NER to extract relevant entities (e.g., game genres, platforms).
3. **Result**: Search results are displayed with relevant details, organized and filtered based on the query.

## 🛠️ Setup and Installation

# IMPORTANT

* **Accessing Results**: This project is unfinished in terms of the UI, to access the results you have to press CTRL + SHIFT + I (Windows/Linux) or Cmd + Option + I (Mac)


### One-Command Setup

To set up and run the entire project, follow these steps:

1. **Install all dependencies** (frontend and backend):
```bash
npm run install-all
```

2. **Start the application**:
```bash
npm start
```


### Within the code itself....
* **Changing between different engines**: You can change between different engines within the main.py by firstly go to line 42 where it states engine_type in the class SearchQuery(). Select your engine from TFIDF, BM25, NER. Then scroll down to line 79-80 and comment out the engines you are not using.
* **Enabling and Disabling of JSON usage**: You can disable or enable the use of JSON when parsing data and building the invertex index within the main.py file by setting html_parser.set_json() -> True/False. (Line 72) TO ACCESS this feature for the HTML Parser
  for the engine (Line 80).
* **Enabling and Disabling of NLP Preprocessing Techniques**: You can disable or enable the use of these NLP techniques such as stemming or lemmatisation using set_use_stemming or set_use_lemmatizer in the main.py. Line(73-74) TO ACCESS THIS FEATURE
* **Enabling and Disabling Query Expansion (Keep disabled without .env file)**: You can disable or enable query expansion by toggling the line of code engine.set_use_json() -> True/False (Line 87)
