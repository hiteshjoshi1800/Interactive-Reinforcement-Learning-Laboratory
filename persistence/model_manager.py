import json
import os
from datetime import datetime


class ModelManager:

    def __init__(self, file_path="data/experiment_history.json"):
        self.file_path = file_path

        os.makedirs(
            os.path.dirname(self.file_path),
            exist_ok=True
        )

        if not os.path.exists(self.file_path):
            self._save([])

    def _load(self):
        with open(self.file_path, "r") as file:
            return json.load(file)

    def _save(self, history):
        with open(self.file_path, "w") as file:
            json.dump(
                history,
                file,
                indent=4
            )

    def save_run(self, run_data):

        history = self._load()

        run_data["run_id"] = len(history) + 1
        run_data["timestamp"] = datetime.now().isoformat()

        history.append(run_data)

        self._save(history)

    def get_history(self):
        return self._load()

    def clear_history(self):
        self._save([])