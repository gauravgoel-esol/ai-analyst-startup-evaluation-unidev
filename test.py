import firebase_admin
from firebase_admin import firestore,credentials
# from langchain_google_vertexai import ChatVertexAI
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from google.oauth2 import service_account

load_dotenv()

# GCP setup
key_path = Path("secrets\ztr1-463311-7921ce5c1de0.json")
project_id = "ztr1-463311"
region = "us-central1"

# Use the application default credentials.
cred = service_account.Credentials.from_service_account_file(key_path)



# Initialize Firebase Admin app (only once)
if not firebase_admin._apps:
    app = firebase_admin.initialize_app(credentials.Certificate(key_path))
else:
    app = firebase_admin.get_app()

db = firestore.client()

print(" Firestore connected successfully!")

# def test_firestore():
#     # Reference to dummy document
#     doc_ref = db.collection("test_collection").document("dummy_doc")

#     # Write data
#     dummy_data = {
#         "message": "Hello Firestore!",
#         "status": "test_ok",
#         "count": 1
#     }
#     doc_ref.set(dummy_data)
#     print(" Dummy document written:", dummy_data)

#     # Read data back
#     doc = doc_ref.get()
#     if doc.exists:
#         print(" Firestore doc retrieved:", doc.to_dict())
#     else:
#         print(" No such document found!")

# if __name__ == "__main__":
#     test_firestore()

def upsert_json_to_firestore(folder_path="data/processed_output", collection_name="VC_data"):
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Folder {folder_path} does not exist.")
        return

    json_files = list(folder.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {folder_path}.")
        return

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Use filename (without extension) as document ID
            doc_id = file_path.stem
            doc_ref = db.collection(collection_name).document(doc_id)

            # Upsert data
            doc_ref.set(data, merge=True)  # merge=True ensures upsert behavior
            print(f"Upserted {doc_id} into collection {collection_name}")

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

def retrieve_vc_data(collection_name="VC_data", doc_id=None):
    """
    Retrieve Firestore documents from VC_data collection.

    Args:
        collection_name (str): Firestore collection name.
        doc_id (str): Optional document ID. If None, retrieves all docs.

    Returns:
        dict: Dictionary of document_id -> document_data
    """
    data = {}

    try:
        if doc_id:
            # Fetch a single document
            doc_ref = db.collection(collection_name).document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                data[doc_id] = doc.to_dict()
            else:
                print(f"Document '{doc_id}' does not exist.")
        else:
            # Fetch all documents in the collection
            docs = db.collection(collection_name).stream()
            for doc in docs:
                data[doc.id] = doc.to_dict()

        return data

    except Exception as e:
        print(f"Error retrieving data: {e}")
        return None

if __name__ == "__main__":
    # upsert_json_to_firestore()
    single_doc = retrieve_vc_data(doc_id="Dr. Doodley Investor Deck Aug 2025_pages")
    print("Single doc:", single_doc)

    # Retrieve all documents in VC_data
    all_docs = retrieve_vc_data()
    print(f"Total docs retrieved: {len(all_docs)}")