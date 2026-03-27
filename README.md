# doc-qa-app

A local document question-answering app built with FastAPI.

## Current version
v0.1: retrieval-based local prototype

## What it does
- upload a document
- ask a question
- retrieve relevant chunks
- generate an answer from those chunks

## Stack
- FastAPI
- LiteLLM
- LangChain
- FAISS

## Current limitation
The app is still slow because the document is processed again on each request.

## Next steps
- process the document once and query it many times
- improve retrieval quality
- deploy online
