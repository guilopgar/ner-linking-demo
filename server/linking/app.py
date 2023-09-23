from flask import Flask, jsonify, request
from tasks import generate_candidates_by_project

import sys
import logging


app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.DEBUG)


@app.route('/generate_candidates_by_project', methods=['POST'])
def generate_candidates_by_project_route():
    app.logger.info('generate_candidates_by_project_route was accessed')
    data = request.get_json()
    preds = data.get('preds')
    k_candidates = data.get('k_candidates')
    
    if not preds:
        return jsonify(
            {"error": "Missing the mention.tsv path."}
        ), 400

    result = generate_candidates_by_project.delay(
        preds,k_candidates
    )
    result.wait()

    if result.successful():
        return jsonify(result.result), 200
    else:
        return jsonify(
            {"error": "An error occurred while generating candidates."}
        ), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
