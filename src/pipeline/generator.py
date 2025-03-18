import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class Generator:
    def __init__(self, model_name="gemini-1.5-pro-latest"):
        """Initialize the response generator."""
        self.model = genai.GenerativeModel(model_name)
        logging.info(f"✅ Generator initialized with model: {model_name}")

    def generate_response(self, query, retrieved_chunks, retrieved_sources):
        """Generates an LLM response based on retrieved knowledge chunks."""
        logging.info(f"📝 Generating response for query: {query}")

        if not retrieved_chunks:
            return "⚠️ No relevant information found."

        # Format retrieved context
        context = "\n\n".join([f"- {chunk}" for chunk in retrieved_chunks])

        # Choose the most relevant source
        source_url = retrieved_sources[0] if retrieved_sources else "https://minecraft.wiki"

        # Construct the prompt
        prompt = f"""
        You are a knowledgeable assistant trained on Minecraft Wiki.
        Answer strictly using the provided context. If you are unable to find direct answer in the provided context,
        If required, structure the answer properly using bullet points, etc. If a crafting recipe is asked and you find it in the context only then, make a grid as mentioned in the context to represent the recipe.

        Context:
        {context}

        Query:
        {query}

        If the context does not contain enough information, If the answer is clearly not present in the context just say "I don't know".
        """

        # Generate the response
        try:
            response = self.model.generate_content(prompt)
            final_response = response.text if response else "⚠️ No response generated."
            if "I don't know" in final_response:
                source_url = "https://minecraft.wiki"
            # Append source link
            if source_url:
                final_response += f"\n\n📌 *Read more at:* [Minecraft Wiki]({source_url})"

            return final_response
        except Exception as e:
            logging.error(f"❌ Error generating response: {e}")
            return "⚠️ Error generating response."
