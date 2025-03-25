from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_similarity,
    answer_correctness
)
from ragas import evaluate
from datasets import Dataset
from typing import Optional, List, Dict, Any
from langchain.llms.base import LLM
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
import requests
import time
import json
import pandas as pd
import os
from openai import OpenAI

from src.log_manager import setup_logger
logger = setup_logger("logs/ragas_eval.log")

my_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

class LMStudioLLM(LLM):
    """LangChain-compatible LLM wrapper for LM Studio"""
    
    api_url: str = "http://localhost:1234/v1/chat/completions"
    max_retries: int = 3
    timeout: int = 5000

    @property
    def _llm_type(self) -> str:
        return "lm_studio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_url": self.api_url,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }

        delay = 1
        for attempt in range(self.max_retries):
            try:
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

class OpenAILLM(LLM):
    """LangChain wrapper for OpenAI models"""
    
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 500
    client: OpenAI = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=kwargs.get('openai_key'))

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return ""

class RagasEvaluator:
    def __init__(self, input_data, use_openai=False, openai_key=None):
        self.data = input_data
        self.use_openai = use_openai
        if use_openai:
            self.llm = OpenAILLM(openai_key=openai_key)
        else:
            self.llm = LMStudioLLM()

    def format_for_ragas(self):
        ragas_data = []
        for row in self.data:
            if not all(key in row for key in ["query", "generated_answer", "ground_truth_answer", "retrieved_chunks"]):
                continue
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
        
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_similarity,
                answer_correctness
            ],
            llm=self.llm,
            embeddings=my_embeddings,
            raise_exceptions=False
        )
        
        self._save_results_json(results)
        return results

    def _save_results_json(self, results):
        output_data = []
        for idx, score in enumerate(results.scores):
            output_data.append({
                **self.data[idx],
                **score
            })
        
        os.makedirs("data/evaluation_results", exist_ok=True)
        with open("data/evaluation_results/ragas_result.json", "w") as f:
            json.dump(output_data, f, indent=2)

    def json_to_excel(self):
        """Convert JSON results to Excel format"""
        try:
            json_path = "data/evaluation_results/ragas_result.json"
            excel_path = "data/evaluation_results/ragas_result.xlsx"
            
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            
            with open(json_path) as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)
            logger.info(f"Saved Excel report to {excel_path}")
        except Exception as e:
            logger.error(f"Excel export failed: {str(e)}")