import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.retriever import Retriever
from src.evaluator.logging import EvaluationLogger
from src.pipeline.generator import Generator  # Import LLM Generator
from rouge_score import rouge_scorer  # ✅ New import for ROUGE-L
import logging
from src.log_manager import setup_logger

# Set up logger for test run
global_logger = setup_logger("logs/retrieval_process.log")
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
        self.logger = EvaluationLogger(eval_type="retrieval")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # ✅ COSINE SIMILARITY METHODS (No Redundant Retrievals)
    def compute_context_precision(self, query, retrieved_chunks):
        """Measures how much of the retrieved chunks are relevant to the query."""
        if not retrieved_chunks:
            # print("⚠️ No retrieved chunks for precision evaluation.")
            global_logger.warning("⚠️ No retrieved chunks for precision evaluation.")
            return 0.0

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)

        precision_score = float(cosine_similarity(query_embedding, retrieved_embedding)[0][0]) * 10  # Scale to 0-10
        # print(f"📊 Context Precision Score (Cosine): {precision_score:.2f}")
        global_logger.info(f"📊 Context Precision Score (Cosine): {precision_score:.2f}")

        result = {"query": query, "context_precision_cosine": precision_score}
        self.logger.log(result)
        return precision_score 

    def compute_context_recall(self, query, ground_truth_answer, retrieved_chunks):
        """Measures whether retrieved chunks contain all the necessary details."""
        if not retrieved_chunks:
            # print("⚠️ No retrieved chunks for recall evaluation.")
            global_logger.warning("⚠️ No retrieved chunks for recall evaluation.")
            return 0.0

        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)

        recall_score = float(cosine_similarity(ground_truth_embedding, retrieved_embedding)[0][0]) * 10  # Scale to 0-10
        # print(f"📊 Context Recall Score (Cosine): {recall_score:.2f}")
        global_logger.info(f"📊 Context Recall Score (Cosine): {recall_score:.2f}")

        result = {"query": query, "context_recall_cosine": recall_score}
        self.logger.log(result)
        return recall_score 

    def compute_retrieval_precision(self, query, ground_truth_answer, retrieved_chunks):
        """Measures how much of the retrieved content is actually relevant."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for precision evaluation.")
            return 0.0

        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)

        precision_score = float(cosine_similarity(retrieved_embedding, ground_truth_embedding)[0][0]) * 10  # Scale to 0-10
        global_logger.info(f"📊 Retrieval Precision Score (Cosine): {precision_score:.2f}")

        result = {"query": query, "retrieval_precision_llm": precision_score}  # Now flattened
        self.logger.log(result)
        return precision_score

    # ✅ LLM-BASED METHODS (No Redundant Retrievals)
    def compute_context_precision_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge how relevant retrieved chunks are to the query."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for LLM evaluation.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval quality.

            Given the following USER QUERY:
            "{query}"

            And the RETRIEVED CHUNKS:
            {retrieved_chunks}

            Evaluate how precisely these retrieved chunks match the USER QUERY.  
            Provide a score from 0 to 10, where:
            - 10 means ALL retrieved chunks are perfectly relevant.
            - 5 means HALF are relevant.
            - 0 means NONE are relevant.
            Respond strictly with a single numeric score (no extra text).
            """
            response = self.generator.model.generate_content(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Context Precision Score (LLM): {score:.2f}")
            return score

        except Exception as e:
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

    def compute_context_recall_with_llm(self, query, ground_truth_answer, retrieved_chunks):
        """Uses LLM to judge how complete the retrieved chunks are."""
        if not retrieved_chunks:
          
            global_logger.warning("⚠️ No retrieved chunks for LLM evaluation.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval completeness.

            Given the GROUND TRUTH ANSWER:
            "{ground_truth_answer}"

            And the RETRIEVED CHUNKS:
            {retrieved_chunks}

            Evaluate how comprehensively these retrieved chunks cover the details present in the ground truth answer.
            Provide a score from 0 to 10, where:
            - 10 means FULL coverage.
            - 5 means partial coverage.
            - 0 means no coverage.
            Respond strictly with a single numeric score (no extra text).
            """
            response = self.generator.model.generate_content(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Context Recall Score (LLM): {score:.2f}")
            return score

        except Exception as e:
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

    def compute_retrieval_precision_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge whether retrieved chunks contain extra unnecessary info."""
        if not retrieved_chunks:

            global_logger.warning("⚠️ No retrieved chunks for LLM evaluation.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval precision and conciseness.

            Given the following USER QUERY:
            "{query}"

            And the RETRIEVED CHUNKS:
            {retrieved_chunks}

            Evaluate how precisely these retrieved chunks provide information relevant to the query WITHOUT including irrelevant details.
            Provide a score from 0 to 10, where:
            - 10 means ONLY relevant information is present.
            - 5 means about HALF the information is irrelevant.
            - 0 means MOSTLY irrelevant.
            Respond strictly with a single numeric score (no extra text).
            """
            response = self.generator.model.generate_content(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Retrieval Precision Score (LLM): {score:.2f}")
            return score

        except Exception as e:
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

    # ✅ CONTEXT OVERLAP SCORE (ROUGE-L)
    def compute_context_overlap(self, query, ground_truth_answer, retrieved_chunks):
        """Measures how much of the ground truth answer is contained in retrieved chunks using ROUGE-L."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for context overlap evaluation.")
            return 0.0

        retrieved_text = " ".join(retrieved_chunks)
        rouge_scores = self.rouge_scorer.score(ground_truth_answer, retrieved_text)
        rouge_l_score = rouge_scores["rougeL"].fmeasure * 10  # Scale to 0-10
        global_logger.info(f"📊 Context Overlap Score (ROUGE-L): {rouge_l_score:.2f}")

        result = {"query": query, "context_overlap_rougeL": rouge_l_score}
        self.logger.log(result)
        return rouge_l_score

    # ✅ NEGATIVE RETRIEVAL CHECK
    def compute_negative_retrieval(self, query, retrieved_chunks, threshold=0.2):
        """Checks how many retrieved chunks are irrelevant to the query."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for negative retrieval evaluation.")
            return 10.0  # If nothing is retrieved, it's fully irrelevant.

        query_embedding = self.model.encode([query], normalize_embeddings=True)
        irrelevant_count = sum(
            1 for chunk in retrieved_chunks
            if cosine_similarity(query_embedding, self.model.encode([chunk], normalize_embeddings=True))[0][0] < threshold
        )

        negative_retrieval_score = (irrelevant_count / len(retrieved_chunks)) * 10  # Scale to 0-10
 
        global_logger.info(f"📊 Negative Retrieval Score: {negative_retrieval_score:.2f}")

        result = {"query": query, "negative_retrieval_cosine": negative_retrieval_score}
        self.logger.log(result)
        return negative_retrieval_score

    # ✅ LLM-BASED CONTEXT OVERLAP SCORE
    def compute_context_overlap_with_llm(self, query, ground_truth_answer, retrieved_chunks):
        """Uses LLM to evaluate how well retrieved chunks match the ground truth answer."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for LLM-based context overlap.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval quality.

            Given the USER QUERY:
            "{query}"

            And the GROUND TRUTH ANSWER:
            "{ground_truth_answer}"

            And the RETRIEVED CHUNKS:
            {retrieved_chunks}

            Evaluate how much of the ground truth answer is present in the retrieved chunks.
            Respond with a score from 0 to 10 (no extra text).
            """
            response = self.generator.model.generate_content(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Context Overlap Score (LLM): {score:.2f}")
            return score

        except Exception as e:
   
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

    # ✅ LLM-BASED NEGATIVE RETRIEVAL CHECK
    def compute_negative_retrieval_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge if retrieved chunks contain irrelevant information."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for LLM-based negative retrieval check.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval relevance.

            Given the USER QUERY:
            "{query}"

            And the RETRIEVED CHUNKS:
            {retrieved_chunks}

            Evaluate how many of the retrieved chunks are completely irrelevant to the query.
            Respond with a score from 0 to 10 (no extra text).
            """
            response = self.generator.model.generate_content(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Negative Retrieval Score (LLM): {score:.2f}")
            return score

        except Exception as e:
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

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

    def _handle_llm_exception(self, e):
        error_message = str(e).lower()
        if "resource has been exhausted" in error_message:
            return "FDTKE"
        else:
            return "Error"