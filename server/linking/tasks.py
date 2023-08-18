from celery import Celery
import subprocess
from pymongo import MongoClient

DB_NAME = "bioMnorm"

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)


@celery_app.task
def generate_candidates_by_project(project_id, gazetteer, k_candidates):
    script_path = "scripts/el_candidate_generation_bi_encoder.py"

    result = subprocess.run(
        ["python", script_path, '-p', project_id, '-g', gazetteer, '-k', k_candidates],
        capture_output=True, text=True
    )

    print("\nSTDOUT", result.stdout)
    print("\nSTDERR", result.stderr)

    if result.returncode != 0:
        return f"Error running script: {result.stderr}"

    # Check successful execution
    client = MongoClient(
        "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.1"
    )
    db = client[DB_NAME]
    collection = db['menciones']

    all_documents = collection.find()

    results = []
    for doc in all_documents:
        doc_id = doc.get('document_id')
        text = doc.get('text')
        codes = doc.get('candidate_codes')
        results.append({"document_id": doc_id, "text": text, "codes": codes})

    return results
