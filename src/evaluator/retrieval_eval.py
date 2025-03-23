import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.retriever import Retriever
from src.evaluator.logging import EvaluationLogger
from src.pipeline.generator import Generator  
import numpy as np
import logging
from src.log_manager import setup_logger
from src.evaluator.evaluation_model import EvaluationModel
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer  
from bert_score import score

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
    
    def __init__(self, retriever : Retriever, generator : Generator,evaluation_model: EvaluationModel, embedding_model="BAAI/bge-base-en"):
        """
        Initializes Retrieval Evaluator with BGE embeddings and LLM.
        """
        self.retriever = retriever
        self.generator = generator
        self.model = SentenceTransformer(embedding_model)
        self.logger = EvaluationLogger(eval_type="retrieval")
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

        self.evaluation_model = evaluation_model

    # Non-llm METHODS (No Redundant Retrievals)
    def compute_context_precision(self, query, retrieved_chunks):
        """Measures how much of the retrieved chunks are relevant to the query."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for precision evaluation.")
            return 0.0

        # Cosine similarity part (unchanged)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        cosine_score = float(cosine_similarity(query_embedding, retrieved_embedding)[0][0]) * 10

        # BM25 computation
        tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
        tokenized_query = query.split()
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Min-max normalization of BM25 scores
        min_score = min(bm25_scores)
        max_score = max(bm25_scores)
        
        if max_score == min_score:
            # Avoid division by zero: if all scores are equal, set normalized score to 10 if they are relevant, else 0.
            normalized_scores = [10 for _ in bm25_scores]
        else:
            normalized_scores = [
                10 * (score - min_score) / (max_score - min_score) for score in bm25_scores
            ]
        
        bm25_avg_score = sum(normalized_scores) / len(normalized_scores)
        
        precision_score = (cosine_score + bm25_avg_score) / 2
        global_logger.info(f"📊 Context Precision Score (Cosine + BM25): {precision_score:.2f}")

        return {
            "cosine_score": round(cosine_score, 2),
            "bm25_score": round(bm25_avg_score, 2),
            "combined_precision_score": round(precision_score, 2)
        }


    def compute_context_recall(self, query, ground_truth_answer, retrieved_chunks):
        """Measures whether retrieved chunks contain all the necessary details."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for recall evaluation.")
            return 0.0

        # Cosine similarity for semantic recall
        retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
        ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)
        recall_score_cosine = float(cosine_similarity(ground_truth_embedding, retrieved_embedding)[0][0]) * 10

        # ROUGE-N for exact overlap (ROUGE-1 for unigrams)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        rouge_scores = scorer.score(ground_truth_answer, " ".join(retrieved_chunks))

        # Retrieve the ROUGE-1 F-measure score (for unigrams)
        rouge_n_score = rouge_scores["rouge1"].fmeasure * 10  # Scaling to 0-10

        # BERTScore for semantic overlap
        P, R, F1 = score([ground_truth_answer], [" ".join(retrieved_chunks)], lang='en')
        bert_score = F1.item() * 10  # Scaling to 0-10

        # Combine Cosine, ROUGE-N, and BERTScore
        recall_score = (recall_score_cosine + rouge_n_score + bert_score) / 3
        global_logger.info(f"📊 Context Recall Score (Cosine + ROUGE + BERTScore): {recall_score:.2f}")
        return recall_score


    def compute_context_precision_chunkwise(self, query, retrieved_chunks, threshold=0.3):
        """
        Chunk-level Context Precision:
        1. For each retrieved chunk, compute its cosine similarity with the query.
        2. Count the fraction of chunks with similarity >= threshold.
        3. Also, compute BM25 scores for each chunk and mark a chunk as relevant if its BM25 score is at least 50% 
        of the maximum BM25 score in the set.
        4. Return:
        - 'chunkwise_cosine_precision': (relevant chunks based on cosine) scaled 0-10.
        - 'chunkwise_bm25_precision': (relevant chunks based on BM25) scaled 0-10.
        - 'combined_precision': the average of the above two fractions, scaled 0-10.
        """
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for chunk-level precision evaluation.")
            return {"chunkwise_cosine_precision": 0.0, "chunkwise_bm25_precision": 0.0, "combined_precision": 0.0}

        import numpy as np
        # Encode the query once
        query_embedding = np.array(self.model.encode([query], normalize_embeddings=True))

        # --- Cosine-based precision ---
        relevant_cosine_count = 0
        for chunk in retrieved_chunks:
            chunk_embedding = np.array(self.model.encode([chunk], normalize_embeddings=True))
            cos_sim = float(cosine_similarity(query_embedding, chunk_embedding)[0][0])
            if cos_sim >= threshold:
                relevant_cosine_count += 1
        cosine_precision_fraction = relevant_cosine_count / len(retrieved_chunks)
        chunkwise_cosine_precision = cosine_precision_fraction * 10.0  # scale to 0-10

        # --- BM25-based precision ---
        from rank_bm25 import BM25Okapi
        tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
        tokenized_query = query.split()
        bm25 = BM25Okapi(tokenized_chunks)
        bm25_scores = list(bm25.get_scores(tokenized_query))  # convert to list for safe processing

        # Define relevance: a chunk is relevant if its BM25 score is at least 50% of the maximum BM25 score
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 0.0
        if max_bm25 == 0:
            bm25_precision_fraction = 0.0
        else:
            relevant_bm25_count = sum(1 for s in bm25_scores if s >= 0.5 * max_bm25)
            bm25_precision_fraction = relevant_bm25_count / len(bm25_scores)
        chunkwise_bm25_precision = bm25_precision_fraction * 10.0

        # --- Combined chunk-level precision ---
        combined_precision_fraction = (cosine_precision_fraction + bm25_precision_fraction) / 2
        combined_precision = combined_precision_fraction * 10.0

        global_logger.info(f"Chunkwise Cosine Precision (0-10): {chunkwise_cosine_precision:.2f}")
        global_logger.info(f"Chunkwise BM25 Precision (0-10): {chunkwise_bm25_precision:.2f}")
        global_logger.info(f"Combined Chunkwise Precision (0-10): {combined_precision:.2f}")

        return {
            "chunkwise_cosine_precision": round(chunkwise_cosine_precision, 2),
            "chunkwise_bm25_precision": round(chunkwise_bm25_precision, 2),
            "combined_precision_score": round(combined_precision, 2)
        }


    def compute_context_recall_chunkwise(self, query, ground_truth_answer, retrieved_chunks, threshold=0.3):
        """
        Chunk-level Context Recall:
        1. Treat the entire ground truth as one claim.
        2. For each retrieved chunk, compute its cosine similarity with the ground truth.
        3. If at least one chunk has a similarity >= threshold, consider the ground truth as 'covered'.
        4. Return 10 if covered, otherwise 0. (For a binary measure.)
        """
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for chunk-level recall evaluation.")
            return 0.0

        gt_embedding = np.array(self.model.encode([ground_truth_answer], normalize_embeddings=True))
        similarities = []
        for chunk in retrieved_chunks:
            chunk_embedding = np.array(self.model.encode([chunk], normalize_embeddings=True))
            cos_sim = float(cosine_similarity(gt_embedding, chunk_embedding)[0][0])
            similarities.append(cos_sim)

        avg_similarity = sum(similarities) / len(similarities)
        recall_score = avg_similarity * 10  # scale to 0–10

        global_logger.info(f"Chunkwise Context Recall (0-10): {recall_score:.2f}")
        return recall_score


    # def compute_retrieval_precision(self, query, ground_truth_answer, retrieved_chunks):
    #     """Measures how much of the retrieved content is actually relevant."""
    #     if not retrieved_chunks:
    #         global_logger.warning("⚠️ No retrieved chunks for precision evaluation.")
    #         return 0.0

    #     retrieved_embedding = self.model.encode([" ".join(retrieved_chunks)], normalize_embeddings=True)
    #     ground_truth_embedding = self.model.encode([ground_truth_answer], normalize_embeddings=True)

    #     precision_score_cosine = float(cosine_similarity(retrieved_embedding, ground_truth_embedding)[0][0]) * 10  # Scale to 0-10

    #     # BM25 for term-based precision
    #     tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
    #     tokenized_query = query.split()
    #     bm25 = BM25Okapi(tokenized_chunks)
    #     bm25_score = bm25.get_scores(tokenized_query)  # BM25 score for each chunk
    #     bm25_avg_score = sum(bm25_score) / len(bm25_score) * 10  # Scale to 0-10
        
    #     # Combine Cosine and BM25
    #     precision_score = (precision_score_cosine + bm25_avg_score) / 2
    #     global_logger.info(f"📊 Retrieval Precision Score (Cosine + BM25): {precision_score:.2f}")
    #     return precision_score

    # ✅ LLM-BASED METHODS (No Redundant Retrievals)
    def compute_context_precision_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge how relevant retrieved chunks are to the query."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for LLM evaluation.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are an expert judge evaluating retrieval quality.

            You are given:
            <query>
            {query}
            </query>

            <retrieved_chunks>
            {retrieved_chunks}
            </retrieved_chunks>

            Rate how precisely these retrieved chunks match the query.

            • Score 10 if all retrieved chunks are perfectly relevant.
            • Score 5 if approximately half of the retrieved chunks are relevant.
            • Score 0 if none of the retrieved chunks are relevant.

            Your response must be strictly a single integer between 0 and 10 with no additional text, punctuation, or explanation.
            """
            response = self.evaluation_model.evaluate(prompt)
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

            You are given:
            <ground_truth_answer>
            {ground_truth_answer}
            </ground_truth_answer>

            <retrieved_chunks>
            {retrieved_chunks}
            </retrieved_chunks>

            Rate how comprehensively these retrieved chunks cover the details of the ground truth answer.

            • Score 10 if the retrieved chunks fully cover all details.
            • Score 5 if only some(partial) details are covered.
            • Score 0 if none of the details are covered.

            Your response must be strictly a single integer between 0 and 10 with no additional text or punctuation.

            If you understand, give output only the score.
            """
            response = self.evaluation_model.evaluate(prompt)
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
            You are an expert judge evaluating **focus and precision** in retrieved information.

            USER QUERY:
            <user query>
            "{query}"
            </user query>

            RETRIEVED CHUNKS:
            <retrieved chunks>
            {retrieved_chunks}
            </retrieved chunks>
            
            Your task is to determine whether the retrieved content stays strictly focused on the user's query.

            - Give a score of 10 if all retrieved chunks ONLY contain relevant information.
            - Score 5 if about half the retrieved chunks include unrelated or unnecessary content.
            - Score 0 if most of the content is irrelevant or off-topic.

            Output should be a SINGLE INTEGER between 0 and 10. No text, no explanation, no punctuation.
            """
            response = self.evaluation_model.evaluate(prompt)
            score = self._parse_llm_score(response)

            global_logger.info(f"📊 Retrieval Precision Score (LLM): {score:.2f}")
            return score

        except Exception as e:
            global_logger.error(f"❌ Error generating response: {str(e)}")
            return self._handle_llm_exception(e)

    # ✅ CONTEXT OVERLAP SCORE (ROUGE-L)
    # def compute_context_overlap(self, query, ground_truth_answer, retrieved_chunks):
    #     """Measures how much of the ground truth answer is contained in retrieved chunks using ROUGE-L."""
    #     if not retrieved_chunks:
    #         global_logger.warning("⚠️ No retrieved chunks for context overlap evaluation.")
    #         return 0.0

    #     retrieved_text = " ".join(retrieved_chunks)
    #     rouge_scores = self.rouge_scorer.score(ground_truth_answer, retrieved_text)
    #     rouge_l_score = rouge_scores["rougeL"].fmeasure * 10  # Scale to 0-10
    #     global_logger.info(f"📊 Context Overlap Score (ROUGE-L): {rouge_l_score:.2f}")

    #     # result = {"query": query, "context_overlap_rougeL": rouge_l_score}
    #     # self.logger.log(result)
    #     return rouge_l_score

    # NEGATIVE RETRIEVAL CHECK
    # def compute_negative_retrieval(self, query, retrieved_chunks, threshold=0.2, bm25_threshold=0.1):
    #     """Detects truly irrelevant chunks using both semantic and lexical similarity."""
    #     if not retrieved_chunks:
    #         global_logger.warning("⚠️ No retrieved chunks for negative retrieval evaluation.")
    #         return 10.0  # All irrelevant if nothing was retrieved

    #     query_embedding = self.model.encode([query], normalize_embeddings=True)

    #     tokenized_chunks = [chunk.split() for chunk in retrieved_chunks]
    #     tokenized_query = query.split()
    #     bm25 = BM25Okapi(tokenized_chunks)

    #     irrelevant_count = 0

    #     for i, chunk in enumerate(retrieved_chunks):
    #         chunk_embedding = self.model.encode([chunk], normalize_embeddings=True)
    #         cosine_sim = cosine_similarity(query_embedding, chunk_embedding)[0][0]
    #         bm25_score = bm25.get_scores(tokenized_query)[i]

    #         if cosine_sim < threshold and bm25_score < bm25_threshold:
    #             irrelevant_count += 1

    #     score = (irrelevant_count / len(retrieved_chunks)) * 10
    #     global_logger.info(f"📊 Negative Retrieval Score (Improved): {score:.2f}")
    #     return score



    # # ✅ LLM-BASED CONTEXT OVERLAP SCORE
    # def compute_context_overlap_with_llm(self, query, ground_truth_answer, retrieved_chunks):
    #     """Uses LLM to evaluate how well retrieved chunks match the ground truth answer."""
    #     if not retrieved_chunks:
    #         global_logger.warning("⚠️ No retrieved chunks for LLM-based context overlap.")
    #         return "Error"  # Fallback if LLM failed

    #     try:
    #         prompt = f"""
    #         You are an expert judge evaluating **wording and phrasing overlap** between the expected answer and retrieved content.

    #         <ground truth answer>
    #         "{ground_truth_answer}"
    #         </ground truth answer>

    #         <retrieved chunks>
    #         {retrieved_chunks}
    #         </retrieved chunks>

    #         Rate how closely the wording, terminology, and phrasing of the retrieved chunks match the ground truth.

    #         - Score **10** if they use the same key terms or nearly identical wording.
    #         - Score **5** if they convey similar ideas but with very different words.
    #         - Score **0** if there's little or no overlap in wording.

    #         Only evaluate the **phrasing**, not whether the information is factually correct.

    #         Respond with a SINGLE INTEGER between 0 and 10. No text, no extra symbols.
    #         """
    #         response = self.evaluation_model.evaluate(prompt)
    #         score = self._parse_llm_score(response)

    #         global_logger.info(f"📊 Context Overlap Score (LLM): {score:.2f}")
    #         return score

    #     except Exception as e:
   
    #         global_logger.error(f"❌ Error generating response: {str(e)}")
    #         return self._handle_llm_exception(e)

    # ✅ LLM-BASED NEGATIVE RETRIEVAL CHECK
    def compute_negative_retrieval_with_llm(self, query, retrieved_chunks):
        """Uses LLM to judge if retrieved chunks contain irrelevant information."""
        if not retrieved_chunks:
            global_logger.warning("⚠️ No retrieved chunks for LLM-based negative retrieval check.")
            return "Error"  # Fallback if LLM failed

        try:
            prompt = f"""
            You are a strict judge identifying irrelevant or junk information in retrieved content.

            USER QUERY:
            "{query}"

            RETRIEVED CHUNKS:
            {retrieved_chunks}

            Count how many of the retrieved chunks are **completely unrelated** to the query — i.e., off-topic, irrelevant, or misleading.

            - Score 0 if all chunks are clearly relevant.
            - Score 5 if about half of the content is off-topic or unrelated.
            - Score 10 if most or all chunks are clearly irrelevant or nonsensical.

            Return a SINGLE INTEGER between 0 and 10. No explanation, no punctuation, just the score.

            """
            response = self.evaluation_model.evaluate(prompt)
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
            return float(response.strip())
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