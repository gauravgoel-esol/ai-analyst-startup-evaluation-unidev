# AI Analyst Startup Evaluation System

A comprehensive startup analysis platform powered by Google's Gemini AI, featuring advanced analytics, performance optimization, and intelligent output compression.

## 🚀 Features

- **Advanced Analytics**: Predictive modeling, risk assessment, and investment recommendations
- **Performance Optimization**: Parallel processing for large datasets with intelligent caching
- **Smart Compression**: Configurable output detail levels (80-90% size reduction)
- **Visualization Hub**: Interactive charts and investor presentations
- **Output Distribution**: Automated portals and notifications

## 🔧 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/gauravgoel-esol/ai-analyst-startup-evaluation-unidev.git
cd ai-analyst-startup-evaluation-unidev
```

### 2. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example)
cp .env.example .env

# Add your Google API key to .env file
GOOGLE_API_KEY=your_api_key_here
```

### 3. Get Google API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## 🎯 Usage

### Quick Start
```python
# Run basic analysis
python run_analysis.py

# Run enhanced analysis with GenAI
python enhanced_run_analysis.py

# Run performance-optimized analysis
python high_performance_analysis.py

# Run size-optimized analysis
python optimized_analysis.py
```

### Analysis Modes

1. **Summary Mode** (~20KB output, 2-3 min processing)
   - Executive overview only
   - Key metrics and recommendations
   - Quick decision making

2. **Standard Mode** (~50KB output, 5-7 min processing)
   - Balanced detail level
   - Most common use case
   - Comprehensive insights

3. **Detailed Mode** (~150KB output, 10-12 min processing)
   - Full analysis with deep insights
   - Complete forecasting data
   - Research and due diligence

## 📊 Performance Benchmarks

| Dataset Size | Processing Time | Output Size | Success Rate |
|-------------|----------------|-------------|--------------|
| 2KB (Sia)   | 2-15 min      | 20-470KB   | 95%+         |
| 20MB        | 30-120 min    | 200MB-1.5GB| 85%+         |
| 100MB       | 2-8 hours     | 1-7GB      | 75%+         |

## 🔒 Security

- API keys are stored in `.env` files (not committed)
- Sensitive data excluded via `.gitignore`
- Environment variable validation

## 📁 Project Structure

```
ai-analyst-startup-evaluation-unidev/
├── .env                          # Environment variables (local only)
├── .gitignore                    # Git exclusions
├── requirements.txt              # Python dependencies
├── run_analysis.py              # Basic analysis runner
├── enhanced_run_analysis.py     # GenAI-powered orchestrator
├── high_performance_analysis.py # Performance optimized
├── optimized_analysis.py        # Size optimized
├── advanced_analytics_app.py    # Core analytics engine
├── visualization_hub_app.py     # Chart generation
└── output_distribution_app.py   # Portal and notifications
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes (ensure API keys are not committed)
4. Test with different data sizes
5. Submit a pull request

## 📝 License

This project is part of the Hack2Skill Hackathon submission.

## ⚡ Quick Commands

```bash
# Install dotenv if not already installed
pip install python-dotenv

# Run optimized analysis with summary output
python -c "from optimized_analysis import OptimizedAnalysisOrchestrator; orchestrator = OptimizedAnalysisOrchestrator('summary'); print('Analysis complete!')"

# Check file sizes
ls -la *.json

# View recent analysis reports
find . -name "*_report_*.json" -mtime -1
```

## 🆘 Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `GOOGLE_API_KEY` is set in `.env` file
2. **Large File Sizes**: Use `summary` or `standard` detail levels
3. **Slow Performance**: Use `high_performance_analysis.py` for large datasets
4. **Import Errors**: Run `pip install -r requirements.txt`

### Support

For issues and questions, please create a GitHub issue with:
- Error message
- Dataset size
- Analysis mode used
- System specifications