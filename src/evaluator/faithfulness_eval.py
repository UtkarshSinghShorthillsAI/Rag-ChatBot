import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer 
from src.pipeline.retriever import Retriever
from src.pipeline.generator import Generator
from src.evaluator.logging import EvaluationLogger
from src.evaluator.evaluation_model import EvaluationModel
from bert_score import score as bert_score

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class FaithfulnessEvaluator:
    """
    Evaluates the faithfulness of LLM-generated responses by checking retrieval consistency and grounding.
    """

    def __init__(self, retriever: Retriever, generator: Generator, evaluation_method : EvaluationModel, embedding_model="BAAI/bge-base-en"):
        """
        Initializes the FaithfulnessEvaluator.
        """
        self.retriever = retriever
        self.generator = generator
        self.logger = EvaluationLogger(eval_type="faithfulness")
        self.model = SentenceTransformer(embedding_model)  # ✅ Now using `bge-base-en`
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)  # ✅ ROUGE-L Scorer

        self.evaluation_method = evaluation_method

    def evaluate_faithfulness(self, query, ground_truth_answer, top_k=5):
        """
        Runs all faithfulness evaluation methods in one go, avoiding redundant retrievals/generations.
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
        blobwise_answer_similarity = self.compute_blobwise_similarity(query, retrieved_chunks, generated_answer)
        chunkwise_answer_similarity = self.compute_chunkwise_similarity(generated_answer, retrieved_chunks)
        faithful_coverage = self.compute_faithful_coverage(query, ground_truth_answer, generated_answer)
        # negative_faithfulness = self.compute_negative_faithfulness(query, retrieved_chunks, generated_answer)

        # ✅ Compute LLM-based faithfulness metrics
        try:
            faithfulness_llm = self.llm_as_judge(query, retrieved_chunks, generated_answer)
            faithful_coverage_llm = self.llm_faithful_coverage(query, ground_truth_answer, generated_answer)
        except Exception as e:
            print(f"❌ Error generating response: {str(e)}")
            faithfulness_llm = "FDTKE"
            faithful_coverage_llm = "FDTKE"

        print("\n📊 Final Faithfulness Evaluation Scores:")
        print(f"✅ Answer-Chunk Blobwise Similarity: {blobwise_answer_similarity:.2f}")
        print(f"✅ Answer-Chunk Chunkwise Similarity: {chunkwise_answer_similarity:.2f}")
        print(f"✅ Faithful Coverage (ROUGE-L): {faithful_coverage:.2f}")
        # print(f"✅ Negative Faithfulness: {negative_faithfulness:.2f}")
        print(f"🤖 Faithfulness Score (LLM): {faithfulness_llm}")
        print(f"🤖 LLM-Based Faithful Coverage: {faithful_coverage_llm}")

        return {
            "query": query,
            "generated_answer": generated_answer,
            "ground_truth_answer": ground_truth_answer,
            "blobwise_answer_similarity": blobwise_answer_similarity,
            "avg_chunkwise_answer_similarity": chunkwise_answer_similarity['avg_chunkwise_score'],
            "max_chunkwise_answer_similarity": chunkwise_answer_similarity['max_chunkwise_score'],
            "faithful_coverage": faithful_coverage,
            # "negative_faithfulness": negative_faithfulness,
            "faithfulness_llm": faithfulness_llm,
            "faithful_coverage_llm": faithful_coverage_llm
        }

    # ✅ NON-LLM BASED METHODS

    def compute_blobwise_similarity(self, query, retrieved_chunks, generated_answer):
        """
        Measures the cosine similarity between the generated answer and the concatenated retrieved chunks.
        """
        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)

        similarity_score = float(cosine_similarity(answer_embedding, retrieved_embedding)[0][0]) * 10  # ✅ Scaled to 0-10

        return similarity_score
    def compute_chunkwise_similarity(self, generated_answer: str, retrieved_chunks: list) -> dict:
        """Computes cosine similarity between the generated answer and each retrieved chunk, returns avg and max."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for chunkwise similarity.")
            return {"avg_chunkwise_score": 0.0, "max_chunkwise_score": 0.0}

        answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)
        chunk_embeddings = self.model.encode(retrieved_chunks, normalize_embeddings=True)

        similarities = cosine_similarity(answer_embedding, chunk_embeddings)[0]
        similarities = [float(sim * 10) for sim in similarities]

        avg_score = sum(similarities) / len(similarities)
        max_score = max(similarities)

        print(f"📊 Chunkwise Avg Similarity (0–10): {avg_score:.2f}")
        print(f"📊 Chunkwise Max Similarity (0–10): {max_score:.2f}")

        return {
            "avg_chunkwise_score": round(avg_score, 2),
            "max_chunkwise_score": round(max_score, 2)
        }


    def compute_faithful_coverage(self, ground_truth_answer: str, generated_answer: str) -> float:
        """Evaluates how much of the ground truth is reflected in the generated answer (ROUGE + BERTScore)."""
        if not ground_truth_answer.strip() or not generated_answer.strip():
            print("⚠️ Empty ground truth or generated answer.")
            return 0.0

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth_answer, generated_answer)
        rouge_l_f1 = rouge_scores["rougeL"].fmeasure * 10  # Scale to 0–10

        # BERTScore
        P, R, F1 = bert_score([generated_answer], [ground_truth_answer], lang='en')
        bert_f1_score = F1.item() * 10  # Scale to 0–10

        # Combined Faithful Coverage Score
        final_score = (rouge_l_f1 + bert_f1_score) / 2

        print(f"📊 Faithful Coverage (ROUGE-L): {rouge_l_f1:.2f}")
        print(f"📊 Faithful Coverage (BERTScore): {bert_f1_score:.2f}")
        print(f"📊 Final Faithful Coverage Score (Avg): {final_score:.2f}")

        return round(final_score, 2)

    # def compute_negative_faithfulness(self, query, retrieved_chunks, generated_answer):
    #     """Checks if the generated content contains **unverified** information not present in retrieved chunks."""
    #     query_embedding = self.model.encode([query], normalize_embeddings=True)
    #     answer_embedding = self.model.encode([generated_answer], normalize_embeddings=True)

    #     negative_score = (1 - cosine_similarity(query_embedding, answer_embedding)[0][0]) * 10  # ✅ Scale to 0-10

    #     return negative_score

    # ✅ LLM-BASED METHODS

    def llm_as_judge(self, query, retrieved_chunks, generated_answer):
        """
        Uses LLM to evaluate **faithfulness of generated response**.
        """
        prompt = f"""
        You are an expert faithfulness evaluator.

        <retrieved context>
        {retrieved_chunks}
        </retrieved context>

        <generated answer>
        {generated_answer}
        </generated answer>

        How **faithful** is the generated answer to the retrieved context?
        
        Provide a score from 0 to 10, where:
        - 10 means the answer is **perfectly faithful** to the retrieved context.
        - 5 means the answer is **somewhat faithful**, but adds **extra information**.
        - 0 means the answer is **completely unfaithful**.

        Respond with a single numeric score (no extra text).
        """
        response = self.evaluation_method.evaluate(prompt)
        return self._parse_llm_score(response)

    def llm_faithful_coverage(self, query, ground_truth_answer, generated_answer):
        """Uses LLM to judge how much of the **ground truth answer** is actually present in the generated response."""
        prompt = f"""
        You are an expert judge evaluating retrieval quality.

        <ground truth answer>
        "{ground_truth_answer}"
        </ground truth answer>

        <generated answer>
        {generated_answer}
        </generated answer>

        How much of the **ground truth answer** is present in the **generated answer**?

        Provide a **score from 0 to 10**, where:
        - 10 means the generated answer **fully contains all the important details** from the ground truth.
        - 5 means it **contains partial details**.
        - 0 means it **contains none of the important details**.

        Respond strictly with a **single numeric score** (no extra text).
        """
        response = self.evaluation_method.evaluate(prompt)
        return self._parse_llm_score(response)

    # ✅ Helper Functions
    def _parse_llm_score(self, response):
        """Extracts numerical score from LLM response."""
        try:
            return float(response.strip())
        except ValueError:
            import re
            match = re.search(r"(\d+(\.\d+)?)", response.text)
            return float(match.group(1)) if match else -1
