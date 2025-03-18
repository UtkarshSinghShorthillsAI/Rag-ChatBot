import os
import logging
import chromadb
from google.generativeai import configure, embed_content
from dotenv import load_dotenv

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
configure(api_key=GEMINI_API_KEY)

class Retriever:
    def __init__(self, db_path="data/vector_db", collection_name="minecraft_wiki"):
        """Initialize the retriever with the vector database."""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        # Check if embeddings exist
        total_embeddings = self.collection.count()
        if total_embeddings == 0:
            logging.warning("⚠️ No embeddings found in ChromaDB! Run embedding first.")
        else:
            logging.info(f"✅ ChromaDB initialized with {total_embeddings} documents.")

    def get_gemini_embedding(self, text):
        """Generate an embedding using Gemini API."""
        try:
            response = embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        except Exception as e:
            logging.error(f"❌ Error generating embedding: {e}")
            return None

    def query(self, query_text, top_k=5):
        """Retrieves relevant chunks based on the query."""
        logging.info(f"🔍 Querying for: {query_text}")

        query_embedding = self.get_gemini_embedding(query_text)
        if not query_embedding:
            logging.error("❌ Failed to generate embedding for query.")
            return [], []

        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        retrieved_chunks = []
        retrieved_sources = []

        if results and "metadatas" in results:
            for metadata in results["metadatas"][0]:
                retrieved_chunks.append(metadata["text"])
                retrieved_sources.append(metadata.get("source", "Unknown Source"))

        logging.info(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks, retrieved_sources
