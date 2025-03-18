import json
import csv
import os

class EvaluationLogger:
    def __init__(self, eval_type="retrieval", json_path=None, csv_path=None):
        self.eval_type = eval_type.lower()
        self.json_path = json_path if json_path else f"data/{self.eval_type}_evaluation.json"
        self.csv_path = csv_path if csv_path else f"data/{self.eval_type}_evaluation.csv"

    def log(self, data):
        """Logs data to JSON only."""
        self.log_to_json(data)

    def log_to_json(self, data):
        """Appends evaluation data to the JSON file while avoiding duplicate queries."""
        existing_data = []
        
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []  # Reset if file is corrupted

        # Avoid duplicate query entries by updating the last entry if it matches the query.
        if existing_data and existing_data[-1].get("query") == data.get("query"):
            existing_data[-1].update(data)
        else:
            existing_data.append(data)

        with open(self.json_path, "w") as f:
            json.dump(existing_data, f, indent=2)

        print(f"✅ Data logged to {self.json_path}")

    def generate_csv(self):
        """Generates a CSV from the JSON log file, handling both retrieval (nested) and faithfulness (flat) formats correctly."""
        if not os.path.exists(self.json_path):
            print("No JSON log file found.")
            return

        with open(self.json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("JSON log file is corrupted.")
                return

        if not data:
            print("JSON log file is empty.")
            return

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # ✅ Differentiate between retrieval and faithfulness logs
            if self.eval_type == "retrieval":  # Retrieval logs (nested)
                self._generate_retrieval_csv(writer, data)
            else:  # Faithfulness logs (flat)
                self._generate_faithfulness_csv(writer, data)

        print(f"✅ CSV successfully generated: {self.csv_path}")

    def _generate_retrieval_csv(self, writer, data):
        """Handles CSV generation for retrieval evaluation logs (nested JSON)."""
        # Determine column headers in order using the first entry as reference.
        first_entry = data[0]
        column_headers = []
        for key, value in first_entry.items():
            if isinstance(value, dict):
                for subkey in value.keys():
                    column_headers.append((key, subkey))
            else:
                column_headers.append((key, None))
        
        # Add any additional columns from subsequent entries.
        for entry in data[1:]:
            for key, value in entry.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        if (key, subkey) not in column_headers:
                            column_headers.append((key, subkey))
                else:
                    if (key, None) not in column_headers:
                        column_headers.append((key, None))
        
        # Write header rows: first row for parent keys and second row for subkeys.
        writer.writerow([key for key, subkey in column_headers])
        writer.writerow([subkey if subkey is not None else "" for key, subkey in column_headers])
        
        # Write data rows.
        for entry in data:
            row = []
            for key, subkey in column_headers:
                if subkey is None:
                    row.append(entry.get(key, ""))
                else:
                    nested = entry.get(key, {})
                    if isinstance(nested, dict):
                        row.append(nested.get(subkey, ""))
                    else:
                        row.append("")
            writer.writerow(row)


    def _generate_faithfulness_csv(self, writer, data):
        """Handles CSV generation for faithfulness evaluation logs (flat JSON)."""
        headers = list(data[0].keys())  # Assume first entry has all keys
        writer.writerow(headers)

        for entry in data:
            writer.writerow([entry.get(col, "") for col in headers])

    def read_last_entry(self):
        """Reads the last logged entry from the JSON file."""
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as f:
                try:
                    data = json.load(f)
                    if data:
                        return data[-1]  # Return the last log entry
                except json.JSONDecodeError:
                    return None
        return None
