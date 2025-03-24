import logging
from pipeline.retriever import Retriever
from pipeline.generator import Generator

from src.log_manager import setup_logger

# Set up logger for test run
logger = setup_logger("logs/pipeline.log")

class RAGPipeline:
    def __init__(self):
        """Initialize the full RAG pipeline."""
        logger.info("🚀 Initializing RAG Pipeline...")

        self.retriever = Retriever()
        self.generator = Generator()

        logger.info("✅ RAG Pipeline initialized!")

    def process_query(self, query):
        """Process a user query through the RAG pipeline."""
        logger.info("\n⏳ Retrieving and Generating Response...\n")
        logger.info(f"📝 User Query: {query}")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks, retrieved_sources = self.retriever.query(query, top_k=5)

        # Step 2: Generate response
        gen_answer, response = self.generator.generate_response(query, retrieved_chunks, retrieved_sources)
        logger.info(f"✅ Response generated for query: {query}")
        return response
