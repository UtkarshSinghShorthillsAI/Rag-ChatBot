import logging
from pipeline.pipeline import RAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info("💡 Welcome to the Minecraft Wiki Chatbot! Type your question below.")
    logging.info("🔍 Type 'exit' to quit.")

    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline()

    while True:
        query = input("\n📝 Your Question: ")
        if query.lower() == "exit":
            logging.info("👋 Exiting the chatbot.")
            break

        response = rag_pipeline.process_query(query)
        print("\n💬 Response:", response)

if __name__ == "__main__":
    main()
