import openai
import os
import requests
import json
import time
from src.log_manager import setup_logger

logger = setup_logger("logs/evaluation_model.log")

class EvaluationModel:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def evaluate(self, query: str, context: list, answer: str) -> dict:
        raise NotImplementedError("Must be implemented by subclass.")


class ChatGPTEvaluationModel(EvaluationModel):
    def __init__(self, api_key: str, model="gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model

    def evaluate(self, prompt: str) -> str:
        retries = 3
        delay = 2

        for attempt in range(retries):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0,
                    timeout=15
                )
                content = response['choices'][0]['message']['content'].strip()
                logger.info(f"[ChatGPT] Prompt succeeded (tokens used: {response['usage']['total_tokens']})")
                return content
            except Exception as e:
                logger.warning(f"[ChatGPT] Attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"[ChatGPT] All retries failed. Final error: {e}")
                    return f"Error: {str(e)}"


class LMStudioEvaluationModel:
    def __init__(self, api_url="http://192.168.1.6:1235/v1/chat/completions"):
        self.api_url = api_url
        # self.model = model

    def evaluate(self, prompt: str) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}]
        }

        retries = 3
        delay = 2

        for attempt in range(retries):
            try:
                response = requests.post(self.api_url, json=payload, timeout=15)
                if response.status_code == 200:
                    content = response.json()['choices'][0]['message']['content'].strip()
                    logger.info("[LMStudio] Prompt succeeded.")
                    return content
                else:
                    logger.warning(f"[LMStudio] Status {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.error(f"[LMStudio] Failed after retries. Response: {response.text}")
                        return f"Error {response.status_code}: {response.text}"
            except Exception as e:
                logger.warning(f"[LMStudio] Attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"[LMStudio] All retries failed. Final error: {e}")
                    return f"Error: {str(e)}"
