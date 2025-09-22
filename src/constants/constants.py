from pathlib import Path
import google.generativeai as genai
import os


#Initialization parameters
key_path = Path("secrets\ztr1-463311-7921ce5c1de0.json")
project_id = "ztr1-463311"
region = "us-central1"

# Firestore parameters
folder_path="data/processed_output"
collection_name="VC_data"

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-pro")

