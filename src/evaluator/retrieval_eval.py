import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.retriever import Retriever
from src.evaluator.logging import EvaluationLogger
from src.pipeline.generator import Generator  # Import LLM Generator

class RetrievalEvaluator:
    """
    Evaluates the retrieval system of the RAG pipeline using:
    - Context Precision
    - Context Recall
    - Retrieval Precision
    - LLM-based versions of the above metrics
    """

    def __init__(self, retriever, generator, embedding_model="BAAI/bge-base-en"):
        """
        Initializes Retrieval Evaluator with BGE embeddings and LLM.
        """
        self.retriever = retriever
        self.generator = generator  # LLM-based evaluation
        self.model = SentenceTransformer(embedding_model)  # ✅ Using `bge-base-en`
        self.logger = EvaluationLogger()

    # ✅ COSINE SIMILARITY METHODS (No Redundant Retrievals)
    def compute_context_precision(self, query, retrieved_chunks):
        """Measures how much of the retrieved chunks are relevant to the query."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for precision evaluation.")
            return 0.0

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)

        precision_score = float(cosine_similarity(query_embedding, retrieved_embedding)[0][0])
        print(f"📊 Context Precision Score (Cosine): {precision_score:.2f}")

        # Log both cosine and LLM scores
        result = {"query": query, "context_precision": {"cosine": precision_score}}
        self.logger.log(result)
        return precision_score

    def compute_context_recall(self, query, ground_truth_answer, retrieved_chunks):
        """Measures whether retrieved chunks contain all the necessary details."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for recall evaluation.")
            return 0.0

        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)

        recall_score = float(cosine_similarity(ground_truth_embedding, retrieved_embedding)[0][0])
        print(f"📊 Context Recall Score (Cosine): {recall_score:.2f}")

        result = {"query": query, "context_recall": {"cosine": recall_score}}
        self.logger.log(result)
        return recall_score

    def compute_retrieval_precision(self, query, ground_truth_answer, retrieved_chunks):
        """Measures how much of the retrieved content is actually relevant."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for precision evaluation.")
            return 0.0

        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)

        precision_score = float(cosine_similarity(retrieved_embedding, ground_truth_embedding)[0][0])
        print(f"📊 Retrieval Precision Score (Cosine): {precision_score:.2f}")

        result = {"query": query, "retrieval_precision": {"cosine": precision_score}}
        self.logger.log(result)
        return precision_score

    # ✅ LLM-BASED METHODS (No Redundant Retrievals)
    def compute_context_precision_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge how relevant retrieved chunks are to the query."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for LLM evaluation.")
            return 0.0

        prompt = f"""
        You are an expert judge evaluating retrieval quality.

        Given the following USER QUERY:
        "{query}"

        And the RETRIEVED CHUNKS:
        {retrieved_chunks}

        Evaluate how precisely these retrieved chunks match the USER QUERY.  
        Provide a score from 0 to 10, where:

        - 10 means ALL retrieved chunks are perfectly relevant to the query.
        - 5 means HALF of the retrieved chunks are relevant, and HALF are irrelevant.
        - 0 means NONE of the retrieved chunks are relevant to the query.

        Respond strictly with a single numeric score (no additional text).
        """
        response = self.generator.model.generate_content(prompt)
        score = self._parse_llm_score(response)

        print(f"📊 Context Precision Score (LLM): {score:.2f}")

        # Log both scores
        result = self._log_with_llm_score("context_precision", query, score)
        self.logger.log(result)
        return score

    def compute_context_recall_with_llm(self, query, ground_truth_answer, retrieved_chunks):
        """Uses LLM to judge how complete the retrieved chunks are."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for LLM evaluation.")
            return 0.0

        prompt = f"""
        You are an expert judge evaluating retrieval completeness.

        Given the GROUND TRUTH ANSWER:
        "{ground_truth_answer}"

        And the RETRIEVED CHUNKS:
        {retrieved_chunks}

        Evaluate how comprehensively these retrieved chunks cover the details present in the GROUND TRUTH ANSWER.

        Provide a score from 0 to 10, where:

        - 10 means retrieved chunks FULLY cover ALL important details from the ground truth answer.
        - 5 means retrieved chunks cover SOME (but not all) key details.
        - 0 means retrieved chunks DO NOT cover ANY of the important details from the ground truth answer.

        Respond strictly with a single numeric score (no additional text).
        """
        response = self.generator.model.generate_content(prompt)
        score = self._parse_llm_score(response)

        print(f"📊 Context Recall Score (LLM): {score:.2f}")

        result = self._log_with_llm_score("context_recall", query, score)
        self.logger.log(result)
        return score

    def compute_retrieval_precision_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge whether retrieved chunks contain extra unnecessary info."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for LLM evaluation.")
            return 0.0

        prompt = f"""
        You are an expert judge evaluating retrieval precision and conciseness.

        Given the following USER QUERY:
        "{query}"

        And the RETRIEVED CHUNKS:
        {retrieved_chunks}

        Evaluate how precisely these retrieved chunks provide information relevant to the query WITHOUT including irrelevant or unnecessary details.

        Provide a score from 0 to 10, where:

        - 10 means retrieved chunks contain ONLY relevant information, with NO unnecessary or irrelevant details.
        - 5 means retrieved chunks contain about HALF unnecessary or irrelevant information.
        - 0 means retrieved chunks contain MOSTLY unnecessary or irrelevant information.

        Respond strictly with a single numeric score (no additional text).
        """
        response = self.generator.model.generate_content(prompt)
        score = self._parse_llm_score(response)

        print(f"📊 Retrieval Precision Score (LLM): {score:.2f}")

        result = self._log_with_llm_score("retrieval_precision", query, score)
        self.logger.log(result)
        return score

    # ✅ Helper Functions
    def _parse_llm_score(self, response):
        """Extracts numerical score from LLM response."""
        try:
            return float(response.text.strip())
        except ValueError:
            import re
            match = re.search(r"(\d+(\.\d+)?)", response.text)
            return float(match.group(1)) if match else -1

    def _log_with_llm_score(self, metric_name, query, llm_score):
        """Adds LLM-based scores to existing logged results."""
        log_entry = {"query": query}
        existing_log = self.logger.read_last_entry()  # Assume logger has a function to read last entry

        if existing_log and metric_name in existing_log:
            existing_log[metric_name]["llm"] = llm_score
            log_entry = existing_log
        else:
            log_entry[metric_name] = {"llm": llm_score}

        return log_entry
