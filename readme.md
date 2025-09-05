# PharmaGPT Backend - AI-Powered Pharmaceutical Quality Analysis

## 🔬 Features

- **Visual Quality Analysis**: GPT-4 Vision powered image analysis for pharmaceutical quality control
- **AI Chat Assistant**: Regulatory expertise for pharmaceutical professionals
- **Investigation Tools**: 5-Why, Fishbone, and CAPA analysis
- **Green Chemistry Advisor**: Process optimization for sustainability
- **Expert Consultations**: Connect with pharmaceutical professionals
- **Regulatory Compliance**: GMP, USFDA, ICH, EDQM, PMDA support

## 🚀 Visual Quality Analysis Types

1. **Laboratory Results**: HPLC, GC, spectroscopy analysis
2. **Product Quality**: Tablet/capsule defect detection
3. **Equipment Readings**: Instrument display analysis
4. **Process Monitoring**: Control chart analysis
5. **Document Analysis**: Regulatory compliance checking
6. **Contamination Assessment**: Foreign particle detection

## 📋 Requirements

- Python 3.11+
- FastAPI
- MongoDB
- Emergent LLM Key (for GPT-4 Vision integration)

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Environment Variables

```bash
MONGO_URL=mongodb://localhost:27017
DB_NAME=pharmgpt
EMERGENT_LLM_KEY=your_emergent_llm_key
STRIPE_API_KEY=your_stripe_key
```

## 🏃‍♂️ Running

```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

## 📊 API Endpoints

- `GET /api/` - Health check
- `GET /api/image-analysis/types` - Get available analysis types
- `POST /api/image-analysis` - Analyze pharmaceutical images
- `GET /api/image-analysis/history/{user_id}` - Get analysis history
- `POST /api/chat` - AI chat assistant
- `GET /api/consultants` - Expert consultants
- `GET /api/troubleshooting-templates` - Investigation templates

## 🎯 Built for Pharmaceutical Professionals

This backend powers the PharmaGPT mobile application, providing AI-powered quality analysis tools for pharmaceutical manufacturing, R&D, and regulatory compliance.
