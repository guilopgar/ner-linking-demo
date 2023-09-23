from flask import Flask, request, jsonify
import requests
import logging
from flask_cors import CORS
import csv


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

NER_API_ENDPOINT = 'http://127.0.0.1:5000/process-text'
LINKING_API_ENDPOINT = 'http://127.0.0.1:8080/generate_candidates_by_project'

@app.route('/process-clinical-text', methods=['POST'])
def process_clinical_text():
    try:
        # Get data from the frontend
        data = request.json
        clinical_text = data.get('clinical_text')
        k_candidates = data.get('k_candidates')

        if not clinical_text or not k_candidates:
            return jsonify({"error": "Both clinical_text and k_candidates are required!"}), 400

        # Send the clinical text to the NER model
        headers = {'Content-Type': 'text/plain'}
        ner_response = requests.post(NER_API_ENDPOINT, data=clinical_text.encode('utf-8'), headers=headers)
        if ner_response.status_code != 200:
            logger.error("NER model returned non-200 status code.")
            return jsonify({"error": "NER model processing failed!"}), 500

        # Assuming the NER model returns the path to the mentions file
        mentions_path = "/data/mentions.tsv"

        # Send the mentions file path and k_candidates to the linking module
        linking_data = {
            'preds': mentions_path,
            'k_candidates': k_candidates
        }
        linking_response = requests.post(LINKING_API_ENDPOINT, json=linking_data)

        if linking_response.status_code != 200:
            logger.error("Linking module returned non-200 status code.")
            return jsonify({"error": "Linking module processing failed!"}), 500

        # Return the combined results to the frontend
        # Read the annotations.tsv and send it as a response
        mentions = []
        with open("../server/data/mentions.tsv", 'r') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            next(reader)  # skip the header row
            for row in reader:
                mentions.append({
                    "label": row[0],
                    "start": int(float(row[1])),
                    "end": int(float(row[2])),
                    "span": row[3],
                    "code": row[4]
                })
        
        response_data = {
            "message": "Text processed successfully",
            "mentions": mentions
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.exception("An error occurred during processing.")
        return jsonify({"error": "An unexpected error occurred!"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
