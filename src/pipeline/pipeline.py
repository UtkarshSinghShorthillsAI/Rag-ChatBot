import logging
from pipeline.retriever import Retriever
from pipeline.generator import Generator

class RAGPipeline:
    def __init__(self):
        """Initialize the full RAG pipeline."""
        logging.info("🚀 Initializing RAG Pipeline...")

        self.retriever = Retriever()
        self.generator = Generator()

        logging.info("✅ RAG Pipeline initialized!")

    def process_query(self, query):
        """Process a user query through the RAG pipeline."""
        logging.info("\n⏳ Retrieving and Generating Response...\n")
        logging.info(f"📝 User Query: {query}")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks, retrieved_sources = self.retriever.query(query, top_k=5)

        # Step 2: Generate response
        response = self.generator.generate_response(query, retrieved_chunks, retrieved_sources)

        return response
