import os
import json
from pathlib import Path
from typing import Dict, List


class FileUtils:
    @staticmethod
    def ensure_dir(directory: str) -> str:
        """Ensure directory exists and return its path"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    @staticmethod
    def save_json(data: Dict, filepath: str):
        """Save data to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Load data from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    @staticmethod
    def write_evaluation(results: Dict, filepath: str):
        """Write evaluation results to file"""
        with open(filepath, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")