import json
import csv
import os

class EvaluationLogger:
    """
    A helper class to log evaluation results to JSON and CSV.
    """

    def __init__(self, json_path="data/evaluation_results.json", csv_path="data/evaluation_results.csv"):
        self.json_path = json_path
        self.csv_path = csv_path

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["query", "context_precision", "context_recall", "retrieval_precision"])

    def log(self, data):
        """
        Logs data to both JSON and CSV files.
        """
        self.log_to_json(data)
        self.log_to_csv(data)

    def log_to_json(self, data):
        """ Appends evaluation data to a JSON file. """
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []  # Handle corrupted/empty JSON file
        else:
            existing_data = []

        existing_data.append(data)

        with open(self.json_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    def log_to_csv(self, data):
        """ Appends evaluation data to a CSV file. """
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                data.get("query", ""),
                data.get("context_precision", ""),
                data.get("context_recall", ""),
                data.get("retrieval_precision", "")
            ])

    def read_last_entry(self):
        """ Reads the last logged entry from the JSON file. """
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    data = json.load(f)
                    if data:
                        return data[-1]  # Return the last entry
                except json.JSONDecodeError:
                    return None  # Handle corrupted JSON
        return None  # Return None if file doesn't exist or is empty
