from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="PharmaGPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"], 
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "PharmaGPT API - Pharmaceutical Quality Assistant",
        "status": "healthy",
        "version": "2.0.0"
    }

@app.get("/api/")  
def api_root():
    return {"message": "PharmaGPT API ready for pharmaceutical assistance"}

@app.post("/api/chat")
def chat_endpoint(request: dict):
    message = request.get("message", "")
    
    response = f"Thank you for your pharmaceutical question: '{message}'. "
    
    if "batch failure" in message.lower():
        response += "For batch failures: 1) Secure the batch, 2) Review documentation, 3) Perform root cause analysis, 4) Implement CAPA."
    elif "gmp" in message.lower():
        response += "For GMP compliance: Ensure proper documentation, equipment calibration, personnel training, and quality systems."
    else:
        response += "I'm here to help with pharmaceutical quality, GMP compliance, and process troubleshooting."
    
    return {
        "response": response,
        "session_id": request.get("session_id", "default"),
        "suggestions": ["Batch failure help", "GMP compliance", "Quality issues"]
    }

@app.get("/api/consultants")
def get_consultants():
    return [
        {
            "id": "1", 
            "name": "Dr. Sarah Johnson",
            "title": "Regulatory Affairs Expert", 
            "rating": 4.9,
            "experience_years": 15
        }
    ]

@app.get("/api/image-analysis/types")
def get_analysis_types():
    return {
        "laboratory_results": {
            "name": "Laboratory Results", 
            "description": "Analyze test results and data"
        }
    }

