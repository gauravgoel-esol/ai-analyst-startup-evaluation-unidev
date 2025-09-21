# Navonmesa – AI-Powered Startup Analyst

![Navonmesa Logo](https://github.com/gauravgoel-esol/ai-analyst-startup-evaluation-unidev/blob/main/data/logo_navonmesa.png)

## Overview

Navonmesa is an AI-driven deal analyst platform that automates startup evaluation by synthesizing founder-provided materials and public data to generate concise, actionable investment insights.

The platform reduces venture capital due diligence time from **weeks to hours** while maintaining institutional-quality insights.

---

## Features

* **Multi-Modal Ingestion**: Upload pitch decks, spreadsheets, call transcripts, emails, and public news for automated parsing and indexing.
* **Generative AI Analysis**: Use Google Cloud’s Gemini/Vertex AI to extract metrics, summarize narratives, and detect patterns (e.g., risk anomalies).
* **Competitive Benchmarking**: Compare startup KPIs (growth, traction, financials) against sector peers using a massive VC dataset.
* **Risk Flagging & Explainability**: Highlights potential risks with annotated explanations.
* **Investor-Ready Summaries**: Generate concise deal memos and slide-ready summaries, customizable by sector or geography.

---

## Architecture (Google Cloud)

* **Cloud Storage & Vision API**: Central repository for raw files; Vision API handles OCR of decks and screenshots.
* **Vertex AI & Gemini Models**: Core analysis engine orchestrates workflows and interprets text/images. Supports custom ML scoring and anomaly detection.
* **BigQuery/Firebase**: Stores structured outputs and public datasets; enables rapid cohort and analytics queries.
* **Firebase Frontend**: Web and mobile interfaces with real-time updates via Firestore; cloud functions trigger analysis jobs.
* **API/Integration**: REST APIs connect to CRM or LP systems; results can be integrated into Salesforce, Affinity, or other tools.

---

## Process Flow (Pipeline)

1. **Document Ingestion**
   Users upload pitch decks, transcripts, and emails. Files are stored in Cloud Storage.

2. **Preprocessing**
   Cloud Vision OCR extracts text from images/PDFs; text is chunked and semantically indexed. Public data is fetched via APIs.

3. **AI Analysis**
   Vertex AI and Gemini LLMs ingest all text/images, extract metrics, and use custom ML models for scoring and anomaly detection.

4. **Benchmarking**
   Structured data is stored in BigQuery and compared against a startup database to compute percentile rankings and peer comparisons.

5. **Insights Generation**
   Vertex AI synthesizes executive summaries, actionable bullet points, and flags risks with explanations.

6. **Output Delivery**
   Frontend displays a dashboard of KPIs, trends, and comparisons. Users can adjust filters or scoring sliders to refine outputs.

---

## Installation (Conda Environment)

```bash
conda create -n aivc python==3.10 -y
conda activate aivc
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Start the FastAPI backend
uvicorn app.main:app --reload

# Access the frontend via Firebase hosting (if configured)
```

---

## Dependencies

* Python 3.10
* FastAPI, Uvicorn
* Pandas, Numpy, Scikit-learn
* Torch, Transformers, Datasets
* Firebase, BigQuery
* Google Cloud Vertex AI & Vision API

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

---

## License

MIT License

---
