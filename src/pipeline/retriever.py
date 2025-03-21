import os
import logging
import chromadb
from sentence_transformers import SentenceTransformer
from src.log_manager import setup_logger

# Set up logger for test run
logger = setup_logger("logs/retriever.log")
# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class Retriever:
    def __init__(self, db_path="data/vector_db", collection_name="minecraft_wiki"):
        """Initialize the retriever with the vector database and embedding model."""
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)

        # Load local embedding model
        self.embedding_model = SentenceTransformer("BAAI/bge-base-en")
        logger.info("🔄 Loaded embedding model: BAAI/bge-base-en")

        # Check if embeddings exist
        total_embeddings = self.collection.count()
        if total_embeddings == 0:
            logger.warning("⚠️ No embeddings found in ChromaDB! Run embedding first.")
        else:
            logger.info(f"✅ ChromaDB initialized with {total_embeddings} documents.")

    def get_embedding(self, text):
        """Generate an embedding using the BGE model."""
        try:
            return self.embedding_model.encode(text, normalize_embeddings=True).tolist()
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return None

    def query(self, query_text, top_k=5):
        """Retrieves relevant chunks based on the query."""
        logger.info(f"🔍 Querying for: {query_text}")

        query_embedding = self.get_embedding(query_text)
        if not query_embedding:
            logger.error("❌ Failed to generate embedding for query.")
            return [], []

        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        
        retrieved_chunks = []
        retrieved_sources = []

        if results and "metadatas" in results:
            for metadata in results["metadatas"][0]:
                retrieved_chunks.append(metadata["text"])
                retrieved_sources.append(metadata.get("source", "Unknown Source"))

        logger.info(f"✅ Retrieved {len(retrieved_chunks)} relevant chunks.")
        logger.info(f"retrieved_chunks : {retrieved_chunks}")
        return retrieved_chunks, retrieved_sources


# Test run (optional)
if __name__ == "__main__":
    retriever = Retriever()
    query_text = "What is a Camel?"
    chunks, sources = retriever.query(query_text)
    print("\n🔍 Query Results:")
    for idx, (chunk, source) in enumerate(zip(chunks, sources)):
        print(f"[{idx+1}] {chunk} (Source: {source})")
