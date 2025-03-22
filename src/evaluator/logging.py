import json
import os
import pandas as pd
from openpyxl import load_workbook
import logging
from src.log_manager import setup_logger

class EvaluationLogger:
    """
    A helper class to log evaluation results to JSON and Excel files in a proper tabular format.
    """

    def __init__(self, eval_type="retrieval", json_path=None, excel_path=None, log_file=None):
        """
        Initializes the logger.
        :param eval_type: "retrieval" or "faithfulness". Determines file naming.
        :param json_path: Optional custom JSON file path.
        :param excel_path: Optional custom Excel file path.
        :param log_file: Path for process tracking logs.
        """
        self.eval_type = eval_type.lower()
        self.json_path = json_path or f"data/evaluation_results/{self.eval_type}_evaluation.json"
        self.excel_path = excel_path or f"data/evaluation_results/{self.eval_type}_evaluation.xlsx"
        # self.log_file = log_file or f"data/evaluation_results/{self.eval_type}_process.log"
        self.logger = setup_logger(f"logs/{self.eval_type}_process.log")

        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)

    def log(self, data):
        """Logs data to JSON and Excel files."""
        self.log_to_json(data)
        self.log_to_process_file(f"Logged data for query: {data.get('query', '')}")
    
    def log_to_json(self, data):
        """Appends evaluation data to the JSON file."""
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []  # Reset if file is corrupted
        else:
            existing_data = []

        # Avoid duplicate query entries
        if existing_data and existing_data[-1].get("query") == data.get("query"):
            existing_data[-1].update(data)
        else:
            existing_data.append(data)

        with open(self.json_path, "w") as f:
            json.dump(existing_data, f, indent=2)

    def log_to_excel(self):
        """Converts JSON log into Excel (one-time batch process)."""
        if not os.path.exists(self.json_path):
            self.log_to_process_file("No JSON data found for writing to Excel.")
            return

        with open(self.json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                self.log_to_process_file("Failed to parse JSON data.")
                return

        df = pd.DataFrame(data)   # Flatten nested JSON properly
        
        # Ensure the correct order of columns: 'query' first, then 'ground_truth_answer', followed by others
        # column_order = ['query', 'ground_truth_answer'] + [col for col in df.columns if col not in ['query', 'ground_truth_answer']]
        # df = df[column_order]

        if os.path.exists(self.excel_path):
            with pd.ExcelWriter(self.excel_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                df.to_excel(writer, index=False, sheet_name="EvaluationResults")
        else:
            df.to_excel(self.excel_path, index=False, sheet_name="EvaluationResults")

        self.log_to_process_file("Successfully wrote evaluation results to Excel.")

    def log_to_process_file(self, message):
        """Writes logs tracking evaluation status."""
        self.logger.info(f"{message}\n")

    def log_error(self, query, error_message):
        """Logs error details when an API or other error occurs."""
        log_entry = {
            "query": query,
            "error": error_message
        }
        self.log_to_json(log_entry)
        self.logger.error(f"Error while processing query '{query}': {error_message}")
