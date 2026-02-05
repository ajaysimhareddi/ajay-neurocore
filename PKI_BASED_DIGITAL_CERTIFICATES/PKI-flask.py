import hashlib
import base64
import requests
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- Google Cloud Platform Configuration ---
# TODO: Replace these placeholder URLs with the actual Trigger URLs of your deployed Google Cloud Functions.
# You can find these on the function's details page in the Google Cloud Console.
SIGN_FUNCTION_URL = " https://us-central1-pki-cert-auth.cloudfunctions.net/sign_and_store"
VERIFY_FUNCTION_URL = "https://us-central1-pki-cert-auth.cloudfunctions.net/verify_signature"


# --- API Endpoints ---

@app.route('/generate_certificate', methods=['POST'])
def generate_certificate():
    """
    Handles certificate upload, generates a hash (Certificate ID), and
    invokes the 'sign_and_store' Google Cloud Function.
    """
    if 'certificate' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['certificate']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        file_content = file.read()
        certificate_id = hashlib.sha256(file_content).hexdigest()

        # --- Trigger Google Cloud Function ---
        print(f"Generated ID: {certificate_id}. Invoking Google Cloud Function for signing.")
        try:
            # Prepare the payload for the Cloud Function.
            # We encode the file content to Base64 to ensure it's sent safely as a JSON string.
            payload = {
                "certificate_id": certificate_id,
                "file_content_b64": base64.b64encode(file_content).decode('utf-8')
            }

            # Make an HTTP POST request to the Cloud Function's trigger URL.
            response = requests.post(SIGN_FUNCTION_URL, json=payload, timeout=30)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            function_result = response.json()

            # Check the response from the function and forward it to the frontend.
            if function_result.get("status") == "success":
                 return jsonify({
                    "message": "Certificate ID Generated and Stored Successfully!",
                    "certificateId": function_result.get("certificateId")
                }), 200
            else:
                return jsonify({"error": function_result.get("message", "An error occurred in the signing function.")}), 500

        except requests.exceptions.RequestException as e:
            print(f"Error calling Google Cloud Function: {e}")
            return jsonify({"error": f"Could not connect to the signing service: {e}"}), 503
        # --- End of Trigger ---

    return jsonify({"error": "An unexpected error occurred"}), 500


@app.route('/verify/<string:certificate_id>', methods=['GET'])
def verify_certificate(certificate_id):
    """
    Handles the public verification request by invoking the 'verify_signature' Google Cloud Function.
    """
    if not all(c in '0123456789abcdef' for c in certificate_id) or len(certificate_id) != 64:
        return jsonify({"error": "Invalid Certificate ID format"}), 400

    # --- Trigger Google Cloud Function ---
    print(f"Received request to verify ID: {certificate_id}. Invoking Google Cloud Function for verification.")
    try:
        # Prepare the payload for the verification function.
        payload = {"certificate_id": certificate_id}

        # Make an HTTP POST request to the Cloud Function's trigger URL.
        response = requests.post(VERIFY_FUNCTION_URL, json=payload, timeout=15)
        response.raise_for_status()

        # The Cloud Function's response is already in the correct format for the frontend.
        verification_result = response.json()
        return jsonify(verification_result), 200

    except requests.exceptions.RequestException as e:
        print(f"Error calling Google Cloud Function: {e}")
        return jsonify({"error": f"Could not connect to the verification service: {e}", "isValid": False}), 503
    # --- End of Trigger ---


# --- Main execution block ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

