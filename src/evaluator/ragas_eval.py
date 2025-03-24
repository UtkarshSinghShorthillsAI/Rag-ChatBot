from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_similarity,
)
from ragas import evaluate
from datasets import Dataset

# LangChain LLM base
from langchain.llms.base import LLM
from typing import Optional, List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
my_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
import requests
import time

# If you have your logger, import it:
from src.log_manager import setup_logger
logger = setup_logger("logs/ragas_eval.log")


class LMStudioLLM(LLM):
    """A LangChain-compatible LLM wrapper around a local LM Studio endpoint."""

    api_url: str = "http://192.168.1.6:1235/v1/chat/completions"
    max_retries: int = 3
    timeout: int = 10

    @property
    def _llm_type(self) -> str:
        """A short name identifying this LLM type."""
        return "lm_studio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return any params that help identify this LLM’s configuration."""
        return {
            "api_url": self.api_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Handle the actual HTTP request to LM Studio and return the final string.
        We’ll reuse existing logic from LMStudioEvaluationModel.
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }

        delay = 1
        for attempt in range(self.max_retries):
            try:
                print("hitting lm studio")
                response = requests.post(self.api_url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    content = response.json()["choices"][0]["message"]["content"].strip()
                    logger.info("[LMStudio] Prompt succeeded.")
                    return content
                else:
                    logger.warning(f"[LMStudio] Status {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(f"[LMStudio] Failed after retries. Response: {response.text}")
                        return f"Error {response.status_code}: {response.text}"
            except Exception as e:
                logger.warning(f"[LMStudio] Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"[LMStudio] All retries failed. Final error: {e}")
                    return f"Error: {str(e)}"


class RagasEvaluator:
    def __init__(self, input_data):
        """Accepts a list of dicts with keys: question, answer, ground_truth_answer, retrieved_chunks"""
        self.data = input_data

    def format_for_ragas(self):
        """Converts internal test data into RAGAS-compatible schema."""
        ragas_data = []
        for row in self.data:
            if not all(key in row for key in ["query", "generated_answer", "ground_truth_answer", "retrieved_chunks"]):
                continue  # skip malformed rows
            ragas_data.append({
                "question": row["query"],
                "answer": row["generated_answer"],
                "ground_truth": row["ground_truth_answer"],
                "contexts": row["retrieved_chunks"]
            })
        return ragas_data

    def run(self):
        ragas_input = self.format_for_ragas()
        dataset = Dataset.from_list(ragas_input)

        print(f"Running RAGAS Evaluation on {len(dataset)} samples...")

        # Create your local LM Studio-based LLM
        my_local_llm = LMStudioLLM()

        results = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_similarity
            ],
            llm=my_local_llm,  # pass your local LLM so RAGAS doesn't try to use OpenAI
            embeddings=my_embeddings
        )
        return results
