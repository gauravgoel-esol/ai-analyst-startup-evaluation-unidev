import os
import json
from pathlib import Path
from src.logging.logger import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import credentials, firestore
from src.components.qualitative_analysis import benchmark_startup, identify_risk_indicators, assess_growth_potential
from src.components.quantitative_analysis import evaluate_startup
from src.components.hybrid_scoring import generate_hybrid_investment_report
from src.components.parser import extract_full_text_from_document 
from src.utils.common import upsert_to_firestore
from src.constants.constants import  collection_name
from src.utils.models import BenchmarkRequest, RiskRequest, GrowthRequest, StartupEvaluationRequest,HybridStartupRequest


app = FastAPI(title="AI Analyst - Document Upload API")


origins = [
    "http://localhost:3000",   
    "http://127.0.0.1:3000",
    "https://navonmesa-frontend-266987433788.us-central1.run.app" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],         
)

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    try:
        logging.info(f"Received file: {file.filename}")
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        pages_dict, total_word_count = extract_full_text_from_document(temp_file_path)

        if not pages_dict:
            os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        output_data = {
            "document_name": file.filename,
            "total_word_count": total_word_count,
            "pages": pages_dict
        }

       
        doc_id = Path(file.filename).stem

        success = upsert_to_firestore(collection_name, doc_id, output_data)
        logging.info(f"Upserted {doc_id} into VC_data")

        os.remove(temp_file_path)

        if success:
            return JSONResponse(content={"status": "success", "doc_id": doc_id, "total_word_count": total_word_count})
        else:
            raise HTTPException(status_code=500, detail="Firestore upsert failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


@app.post("/benchmark_startup")
def benchmark_startup_api(req: BenchmarkRequest):
    try:
        result = benchmark_startup(req.document_text, req.sector)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/identify_risks")
def identify_risks_api(req: RiskRequest):
    try:
        result = identify_risk_indicators(req.document_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assess_growth")
def assess_growth_api(req: GrowthRequest):
    try:
        result = assess_growth_potential(req.document_text, req.investor_weights)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/evaluate_startup")
def evaluate_startup_api(req: StartupEvaluationRequest):
    try:
        result = evaluate_startup(req.startup_data, req.weights)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
@app.post("/analyze_startup")
def analyze_startup(req: HybridStartupRequest):
    try:
        report = generate_hybrid_investment_report(
            req.structured_data,
            req.unstructured_text,
            req.weights
        )
        return report
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")