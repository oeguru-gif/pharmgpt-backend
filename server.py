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
    return {"message": "PharmaGPT API is running successfully!"}

@app.get("/api/")
def api_root():
    return {"message": "PharmaGPT API - Ready for pharmaceutical assistance"}

@app.get("/api/image-analysis/types")
def get_analysis_types():
    return {
        "laboratory_results": {
            "name": "Laboratory Results Analysis",
            "description": "Analyze test results and analytical data"
        },
        "product_quality": {
            "name": "Product Quality Assessment", 
            "description": "Evaluate product appearance and defects"
        }
    }

@app.get("/api/consultants")
def get_consultants():
    return [
        {
            "id": "1",
            "name": "Dr. Sarah Johnson",
            "title": "Regulatory Affairs Expert",
            "experience_years": 15,
            "rating": 4.9
        }
    ]

@app.post("/api/chat")
def chat_basic(request: dict):
    return {
        "response": "Thank you for your pharmaceutical question. This is a basic response while we set up full AI capabilities.",
        "session_id": request.get("session_id", "default"),
        "suggestions": ["Check batch records", "Review quality metrics"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
