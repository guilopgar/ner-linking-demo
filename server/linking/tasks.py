from celery import Celery
import subprocess
from pymongo import MongoClient


SCRIPT_PATH = "scripts/mentions_normalization.py"

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)


@celery_app.task
def generate_candidates_by_project( preds,k_candidates):
    script_path = "scripts/mentions_normalization.py"

    result = subprocess.run(
        ["python", script_path, '-p', preds ,  '-k', k_candidates],
        capture_output=True, text=True
    )

    print("\nSTDOUT", result.stdout)
    print("\nSTDERR", result.stderr)

    if result.returncode != 0:
        return f"Error running script: {result.stderr}"