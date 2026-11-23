# rag-document-qa

A small document question-answering pipeline built in three stages. Each stage adds one idea, so you can see how retrieval-augmented generation (RAG) works.

## How it works

1. Word matching (search.py): splits text into sentences and scores each one by counting shared words. Stop words and simple plural handling are removed first.
2. Semantic search (search_embeddings.py): converts sentences and questions into vectors with Sentence Transformers (all-MiniLM-L6-v2), then ranks them by cosine similarity. This finds matches even when the words differ, such as "vote" and "voters".
3. Grounded answers (answer.py): sends the top matching sentences to the Gemini API and asks it to answer only from those passages.

## Setup

Install the libraries:

py -m pip install sentence-transformers google-genai

Set your Gemini API key as an environment variable. Never put the key in the code:

$env:GEMINI_API_KEY = "your-key-here"

## Run

py search.py              (word matching)
py search_embeddings.py   (semantic search)
py answer.py              (full pipeline with Gemini)

Sample questions for the files in docs/:
- Who can vote in a poll?
- What does Cryptoverse show?

## Limitations

- Searches only .txt files in docs/, one sentence per chunk.
- Always returns the top 2 results, even weak matches. A minimum score threshold is planned.
- Gemini model names change often. Update the model= line in answer.py if you see a 404 error.
