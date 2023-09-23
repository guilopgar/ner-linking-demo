from flask import Flask, request, jsonify
import os
import subprocess
import logging

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

DATA_DIR = "/data"
TEXT_FILE = "text.txt"
SCRIPT_PATH = "scripts/mentions_detection.py"

@app.route('/process-text', methods=['POST'])
def process_text():
    # Log the incoming request
    logging.info(f"Received request with data: {request.data[:100]}...")  # Log only the first 100 chars for brevity

    # Get the raw clinical text from the request
    raw_text = request.data.decode('utf-8', errors='replace')

    # Save the text to a file
    try:
        with open(os.path.join(DATA_DIR, TEXT_FILE), 'w') as f:
            f.write(raw_text)
    except Exception as e:
        logging.error(f"Error while writing to file: {str(e)}")
        return jsonify({"error": f"Error writing to file: {str(e)}"}), 500

  # Run the mentions detection script
    cmd = ["python", SCRIPT_PATH, "-d", DATA_DIR]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    stderr_content = stderr.decode('utf-8')

    if "error" in stderr_content.lower():
        logging.error(f"Error in mentions detection script: {stderr_content}")
        return jsonify({"error": stderr_content}), 500
    elif stderr_content:
        logging.warning(f"Warning from mentions detection script: {stderr_content}")
        return jsonify({"message": "Text processed with warnings.", "warning": stderr_content}), 200

    # For now, just return a success response
    return jsonify({"message": "Text processed successfully"}), 200
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)  # Enable debug mode
