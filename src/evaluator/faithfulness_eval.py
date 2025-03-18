import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer  # ✅ New import for ROUGE-L
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.logging import EvaluationLogger

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class FaithfulnessEvaluator:
    """
    Evaluates the faithfulness of LLM-generated responses by checking retrieval consistency and grounding.
    """

    def __init__(self, retriever: Retriever, generator: Generator, embedding_model="BAAI/bge-base-en"):
        """
        Initializes the FaithfulnessEvaluator.
        """
        self.retriever = retriever
        self.generator = generator
        self.logger = EvaluationLogger(eval_type="faithfulness")
        self.model = SentenceTransformer(embedding_model)  # ✅ Now using `bge-base-en`
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # ✅ ROUGE-L Scorer

    def evaluate_faithfulness(self, query, ground_truth_answer, top_k=5):
        """
        ✅ Runs all faithfulness evaluation methods in one go, avoiding redundant retrievals/generations.
        """
        print(f"\n🔍 Evaluating Faithfulness for Query: {query}")

        # ✅ Retrieve chunks ONCE
        retrieved_chunks, _ = self.retriever.query(query, top_k=top_k)
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks found.")
            return None

        # ✅ Generate answer ONCE
        generated_answer = self.generator.generate_response(query, retrieved_chunks, [])

        # ✅ Compute faithfulness metrics
        answer_chunk_similarity = self.answer_chunk_similarity(query, retrieved_chunks, generated_answer)
        faithful_coverage = self.compute_faithful_coverage(query, ground_truth_answer, generated_answer)
        negative_faithfulness = self.compute_negative_faithfulness(query, retrieved_chunks, generated_answer)

        # ✅ Compute LLM-based faithfulness metrics
        faithfulness_llm = self.llm_as_judge(query, retrieved_chunks, generated_answer)
        faithful_coverage_llm = self.llm_faithful_coverage(query, ground_truth_answer, generated_answer)

        print("\n📊 Final Faithfulness Evaluation Scores:")
        print(f"✅ Answer-Chunk Similarity: {answer_chunk_similarity:.2f}")
        print(f"✅ Faithful Coverage (ROUGE-L): {faithful_coverage:.2f}")
        print(f"✅ Negative Faithfulness: {negative_faithfulness:.2f}")
        print(f"🤖 Faithfulness Score (LLM): {faithfulness_llm:.2f}")
        print(f"🤖 LLM-Based Faithful Coverage: {faithful_coverage_llm:.2f}")

        return {
            "query": query,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "answer_chunk_similarity": answer_chunk_similarity,
            "faithful_coverage": faithful_coverage,
            "negative_faithfulness": negative_faithfulness,
            "faithfulness_llm": faithfulness_llm,
            "faithful_coverage_llm": faithful_coverage_llm
        }

    # ✅ NON-LLM BASED METHODS

    def answer_chunk_similarity(self, query, retrieved_chunks, generated_answer):
        """
        Measures the cosine similarity between the generated answer and the concatenated retrieved chunks.
        """
        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)

        similarity_score = float(cosine_similarity(answer_embedding, retrieved_embedding)[0][0]) * 10  # ✅ Scaled to 0-10

        return similarity_score

    def compute_faithful_coverage(self, query, ground_truth_answer, generated_answer):
        """Measures how much of the **ground truth answer** is contained in the generated response using ROUGE-L."""
        rouge_scores = self.rouge_scorer.score(ground_truth_answer, generated_answer)
        coverage_score = rouge_scores["rougeL"].fmeasure * 10  # ✅ Scale to 0-10

        return coverage_score

    def compute_negative_faithfulness(self, query, retrieved_chunks, generated_answer):
        """Checks if the generated content contains **unverified** information not present in retrieved chunks."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)

        negative_score = (1 - cosine_similarity(query_embedding, answer_embedding)[0][0]) * 10  # ✅ Scale to 0-10

        return negative_score

    # ✅ LLM-BASED METHODS

    def llm_as_judge(self, query, retrieved_chunks, generated_answer):
        """
        Uses LLM to evaluate **faithfulness of generated response**.
        """
        prompt = f"""
        You are an expert evaluator.

        Given the RETRIEVED CONTEXT:
        {retrieved_chunks}

        And the GENERATED ANSWER:
        {generated_answer}

        How **faithful** is the generated answer to the retrieved context?
        
        Provide a score from 0 to 10, where:
        - 10 means the answer is **perfectly faithful** to the retrieved context.
        - 5 means the answer is **somewhat faithful**, but adds **extra information**.
        - 0 means the answer is **completely unfaithful**.

        Respond with a single numeric score (no extra text).
        """
        response = self.generator.model.generate_content(prompt)
        return self._parse_llm_score(response)

    def llm_faithful_coverage(self, query, ground_truth_answer, generated_answer):
        """Uses LLM to judge how much of the **ground truth answer** is actually present in the generated response."""
        prompt = f"""
        You are an expert judge evaluating retrieval quality.

        Given the USER QUERY:
        "{query}"

        And the GROUND TRUTH ANSWER:
        "{ground_truth_answer}"

        And the GENERATED ANSWER:
        {generated_answer}

        How much of the **ground truth answer** is present in the **generated answer**?

        Provide a **score from 0 to 10**, where:
        - 10 means the generated answer **fully contains all the important details** from the ground truth.
        - 5 means it **contains partial details**.
        - 0 means it **contains none of the important details**.

        Respond strictly with a **single numeric score** (no extra text).
        """
        response = self.generator.model.generate_content(prompt)
        return self._parse_llm_score(response)

    # ✅ Helper Functions
    def _parse_llm_score(self, response):
        """Extracts numerical score from LLM response."""
        try:
            return float(response.text.strip())
        except ValueError:
            import re
            match = re.search(r"(\d+(\.\d+)?)", response.text)
            return float(match.group(1)) if match else -1
