import weaviate
from weaviate.classes.query import Filter, MetadataQuery
import requests
from datetime import datetime, timezone
import json
from tqdm import tqdm
import os
from pathlib import Path
from openai import OpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder


model = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

HF_TOKEN = os.environ["HF_TOKEN"]
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def load_article(url: str):
    loader = WebBaseLoader(web_paths=(url,))
    docs = loader.load()
    title = docs[0].metadata.get("title", "Unknown Title")

    return docs, title


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)
    return splits


def prepare_chunks(splits, title):
    chunks = []

    for i, doc in enumerate(splits):
        chunks.append({
            "text": doc.page_content,
            "chunk_id": i,
            "source": doc.metadata["source"],
            "title": title
        })

    return chunks


def embed_chunks(chunks):
    embeddings = []

    for chunk in chunks:
        vector = model.encode(chunk["text"]).tolist()
        embeddings.append(vector)

    return embeddings


def insert_chunks(paragraphs_collection, chunks, embeddings):

    for chunk, vector in zip(chunks, embeddings):

        paragraphs_collection.data.insert(
            properties={
                "text": chunk["text"],
                "para_id": chunk["chunk_id"],
                "doc_id": chunk["source"],
                "title": chunk["title"]
            },
            vector=vector
        )

def document_exists(paragraphs_collection, url):

    response = paragraphs_collection.query.fetch_objects(
        filters=Filter.by_property("doc_id").equal(url),
        limit=1
    )

    return len(response.objects) > 0

def clean_documents(docs):

    cleaned = []

    for doc in docs:

        text = doc.page_content

        text = text.replace("\n\n\n", "\n\n")
        text = text.strip()

        doc.page_content = text
        cleaned.append(doc)

    return cleaned

def embed_query(question):

    vector = model.encode(question).tolist()

    return vector

def retrieve_chunks(paragraphs_collection, query, limit=3):
    query_vector = embed_query(query)
    
    response = paragraphs_collection.query.hybrid(
        query=query,
        vector=query_vector,
        limit=limit,
        alpha=0.5,
        return_metadata=MetadataQuery(score=True)
    )

    return response.objects

def build_context(reranked):

    context_parts = []

    for obj in reranked:
        item = obj["object"]
        paragraph_id = item.properties["para_id"]
        text = item.properties["text"]
        title = item.properties["title"]

        context_parts.append(
            f"[{title} — Paragraph {paragraph_id}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return context

def ask_llm(context, question):

    prompt_template = Path("prompts/qa_prompt.txt").read_text()
    prompt = prompt_template.format(
        context=context,
        question=question
    )

    response = open_ai_client.chat.completions.create(
        model="openai/gpt-oss-120b:groq",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You answer questions strictly using the provided context and cite sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content

def rerank_chunks(question, results, top_k=3):

    pairs = []

    for obj in results:
        text = obj.properties["text"]
        pairs.append((question, text))

    scores = reranker.predict(pairs)

    scored = []
    for obj, score in zip(results, scores):
        scored.append({
            "object": obj,
            "reranker_score": float(score),
            "hybrid_score": obj.metadata.score
        })
    scored.sort(key=lambda x: x["reranker_score"], reverse=True)
    return scored[:top_k]

def ingest_url(url: str):
    if document_exists(paragraphs, url):
        print("Document already ingested. Skipping.")
        return

    docs, title = load_article(url)

    docs = clean_documents(docs)

    splits = chunk_documents(docs)

    chunks = prepare_chunks(splits, title)

    embeddings = embed_chunks(chunks)

    insert_chunks(paragraphs, chunks, embeddings)

    print(f"Inserted {len(chunks)} chunks")

headers = {
    "X-Cohere-Api-Key": COHERE_API_KEY
}  # Replace with your Cohere API key

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=WEAVIATE_API_KEY,
    headers=headers
)

open_ai_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


paragraphs = client.collections.get("Paragraphs")

# -------- Run ingestion --------
if __name__ == "__main__":

    url = "https://arpitbhayani.me/blogs/bm25"

    ingest_url(url)

    question = "What are the characteristics of a hybrid search?"

    results = retrieve_chunks(paragraphs, question, limit=10)
    
    reranked = rerank_chunks(question, results, top_k=3)

    for item in reranked:
        obj = item["object"]
        print("Title:", obj.properties["title"])
        print("Para:", obj.properties["para_id"])
        print("Hybrid Score:", item["hybrid_score"])
        print("Reranker Score:", item["reranker_score"])
        print("----")
    
    context = build_context(reranked)

    answer = ask_llm(context, question)

    print(answer)
    # config = paragraphs.config.get()
    # print(config)

    client.close()
