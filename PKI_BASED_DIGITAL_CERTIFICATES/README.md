# A PKI-Based Approach to Digital Certificates

[](https://github.com/)
[](https://github.com/)
[](https://github.com/)

A secure digital certificate verification system for universities using Public Key Infrastructure (PKI) and Digital Signatures, hosted on Oracle Cloud Infrastructure (OCI). [cite\_start]This project ensures certificate authenticity, integrity, and non-repudiation, preventing forgery and reducing manual verification time[cite: 9, 10].

[cite\_start]Developed for the Cloud Computing (B.Tech, 5th Semester) project at Woxsen University, School of Technology[cite: 1, 2, 3].

-----

## Team Members

  * **S. [cite\_start]Ajay Simha Reddy** – `23WU0102172` [cite: 7]
  * **V. [cite\_start]Vineela** – `23WU0102222` [cite: 7]
  * **Y. [cite\_start]Nuthan** – `23WU0102229` [cite: 7]
  * **V. [cite\_start]Narasimha** – `23WU0102219` [cite: 7]

-----

## System Architecture

[cite\_start]The system is built on a client-server model leveraging Oracle Cloud's robust services[cite: 28]. The workflow is as follows:

1.  [cite\_start]**Generation:** A Python Flask backend generates digital certificates[cite: 37].
2.  [cite\_start]**Signing:** The certificate's hash (its unique ID) is signed using a secure asymmetric key managed by **Oracle Key Management Service (KMS)**[cite: 29].
3.  [cite\_start]**Storage:** The signed certificate is uploaded to a private **Oracle Object Storage** bucket[cite: 29]. The signature is stored as metadata.
4.  **Verification:** A public web portal allows users to enter a Certificate ID. [cite\_start]This triggers an **Oracle Function** which retrieves the certificate and uses KMS to verify its signature against the public key[cite: 30].
5.  [cite\_start]**Access Control:** All cloud resources are protected using granular **OCI IAM** policies to prevent unauthorized access[cite: 33].

-----

## Technology Stack

| Category         | Technology / Service                                      |
| ---------------- | --------------------------------------------------------- |
| **Backend** | Python (Flask)                                            |
| **Frontend** | HTML, CSS, JavaScript                                     |
| **Cloud Provider** | [cite\_start]Oracle Cloud Infrastructure (OCI) [cite: 12]              |
| **Security** | [cite\_start]Oracle Key Management Service (KMS) [cite: 29]            |
| **Storage** | [cite\_start]Oracle Object Storage [cite: 29]                          |
| **Compute** | [cite\_start]Oracle Functions (Serverless) [cite: 30]                  |
| **Hosting** | [cite\_start]OCI Web Application Hosting [cite: 31]                    |
| **CI/CD** | [cite\_start]GitHub [cite: 39]                                         |

-----

## Key Features

  * **Secure Certificate Generation:** Creates tamper-proof digital certificates.
  * **Instant Verification:** A public portal for employers and institutions to verify certificate authenticity in real-time.
  * **Cloud-Native & Scalable:** Built entirely on OCI for high availability and scalability.
  * **Automated Workflows:** Serverless functions handle signing and verification processes automatically.

-----

## Setup and Installation

To run this project locally, follow these steps:

1.  **Prerequisites:**

      * Python 3.8+
      * An active Oracle Cloud Infrastructure (OCI) account.
      * OCI CLI configured on your local machine.

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

3.  **Set Up a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your OCI credentials and resource OCIDs.

    ```env
    OCI_USER_OCID="..."
    OCI_TENANCY_OCID="..."
    OCI_FINGERPRINT="..."
    OCI_PRIVATE_KEY_PATH="..."
    KMS_KEY_OCID="..."
    BUCKET_NAME="..."
    ```

6.  **Run the Flask Application:**

    ```bash
    flask run
    ```

    The application will be available at `http://127.0.0.1:5000`.

-----

## Performance Metrics

The system was tested for reliability and efficiency, achieving the following results:

  * [cite\_start]**Verification Time:** Below 8 seconds on average[cite: 43].
  * [cite\_start]**System Uptime:** 99.8% under concurrent load tests[cite: 45].
  * [cite\_start]**Cost-Effectiveness:** OCI's pricing model proved suitable for institutional-scale deployment[cite: 44].

-----

## Ethical & Sustainability Aspects

  * [cite\_start]**Data Security:** All personal and academic data is encrypted at rest and in transit to ensure privacy and security[cite: 48].
  * **Sustainable Computing:** The project leverages Oracle Cloud’s energy-efficient data centers. [cite\_start]The serverless architecture minimizes idle compute usage, reducing the overall carbon footprint[cite: 49, 50].

-----

## Future Scope

Future enhancements for this project include:

  * [cite\_start]**Blockchain Integration:** To provide an immutable, decentralized ledger for certificate records[cite: 54].
  * [cite\_start]**AI-Based Fraud Detection:** To proactively identify and flag suspicious verification patterns[cite: 54].
  * [cite\_start]**Multi-Institutional Networks:** To create a federated system for cross-institutional verification[cite: 55].
