import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.retriever import Retriever
from src.evaluator.logging import EvaluationLogger
from src.pipeline.generator import Generator  # Import LLM Generator
from rouge_score import rouge_scorer  # ✅ New import for ROUGE-L

class RetrievalEvaluator:
    """
    Evaluates the retrieval system of the RAG pipeline using:
    - Context Precision
    - Context Recall
    - Retrieval Precision
    - Context Overlap Score (ROUGE-L)
    - Negative Retrieval Check
    - LLM-based versions of the above metrics
    """
    
    def __init__(self, retriever, generator, embedding_model="BAAI/bge-base-en"):
        """
        Initializes Retrieval Evaluator with BGE embeddings and LLM.
        """
        self.retriever = retriever
        self.generator = generator
        self.model = SentenceTransformer(embedding_model)
        self.logger = EvaluationLogger()
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

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
        return precision_score*10

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
        return recall_score*10


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
        Provide a score from 0 to 10 can be float, where:

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

        Provide a score from 0 to 10 can be float, where:

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

        Provide a score from 0 to 10 can be float, where:

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


    # ✅ CONTEXT OVERLAP SCORE (ROUGE-L)
    def compute_context_overlap(self, query, ground_truth_answer, retrieved_chunks):
        """Measures how much of the ground truth answer is contained in retrieved chunks using ROUGE-L."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for context overlap evaluation.")
            return 0.0

        retrieved_text = " ".join(retrieved_chunks)
        rouge_scores = self.rouge_scorer.score(ground_truth_answer, retrieved_text)
        rouge_l_score = rouge_scores["rougeL"].fmeasure  # ROUGE-L F1 Score

        print(f"📊 Context Overlap Score (ROUGE-L): {rouge_l_score:.2f}")

        result = {"query": query, "context_overlap": {"rougeL": rouge_l_score}}
        self.logger.log(result)
        return rouge_l_score*10

    # ✅ NEGATIVE RETRIEVAL CHECK
    def compute_negative_retrieval(self, query, retrieved_chunks, threshold=0.2):
        """Checks how many retrieved chunks are irrelevant to the query."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for negative retrieval evaluation.")
            return 1.0  # If nothing is retrieved, it's fully irrelevant.

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        irrelevant_count = sum(
            1
            for chunk in retrieved_chunks
            if cosine_similarity(query_embedding, self.model.encode([chunk], normalize_embeddings=True))[0][0]
            < threshold
        )

        negative_retrieval_score = irrelevant_count / len(retrieved_chunks)  # % of irrelevant chunks
        print(f"📊 Negative Retrieval Score: {negative_retrieval_score:.2f}")

        result = {"query": query, "negative_retrieval": negative_retrieval_score}
        self.logger.log(result)
        return negative_retrieval_score*10

    # ✅ LLM-BASED CONTEXT OVERLAP SCORE
    def compute_context_overlap_with_llm(self, query, ground_truth_answer, retrieved_chunks):
        """Uses LLM to evaluate how well retrieved chunks match the ground truth answer."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for LLM-based context overlap.")
            return 0.0

        prompt = f"""
        You are an expert judge evaluating retrieval quality.

        Given the USER QUERY:
        "{query}"

        And the GROUND TRUTH ANSWER:
        "{ground_truth_answer}"

        And the RETRIEVED CHUNKS:
        {retrieved_chunks}

        Evaluate how much of the GROUND TRUTH ANSWER is present in the RETRIEVED CHUNKS.

        Provide a score from 0 to 10 can be float, where:
        - 10 means retrieved chunks contain the FULL answer exactly as it appears in ground truth.
        - 5 means retrieved chunks contain HALF the important information.
        - 0 means retrieved chunks contain NOTHING relevant to the answer.

        Respond strictly with a single numeric score (no extra text).
        """
        response = self.generator.model.generate_content(prompt)
        score = self._parse_llm_score(response)

        print(f"📊 Context Overlap Score (LLM): {score:.2f}")

        result = self._log_with_llm_score("context_overlap", query, score)
        self.logger.log(result)
        return score

    # ✅ LLM-BASED NEGATIVE RETRIEVAL CHECK
    def compute_negative_retrieval_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge if retrieved chunks contain irrelevant information."""
        if not retrieved_chunks:
            print("⚠️ No retrieved chunks for LLM-based negative retrieval check.")
            return 10.0  # If nothing is retrieved, it's fully irrelevant.

        prompt = f"""
        You are an expert judge evaluating retrieval relevance.

        Given the USER QUERY:
        "{query}"

        And the RETRIEVED CHUNKS:
        {retrieved_chunks}

        Evaluate how many of the retrieved chunks are **completely irrelevant** to the query.

        Provide a score from 0 to 10 can be float, where:
        - 10 means **ALL retrieved chunks are irrelevant**.
        - 5 means **HALF of the retrieved chunks are irrelevant**.
        - 0 means **ALL retrieved chunks are relevant**.

        Respond strictly with a single numeric score (no extra text).
        """
        response = self.generator.model.generate_content(prompt)
        score = self._parse_llm_score(response)

        print(f"📊 Negative Retrieval Score (LLM): {score:.2f}")

        result = self._log_with_llm_score("negative_retrieval", query, score)
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
