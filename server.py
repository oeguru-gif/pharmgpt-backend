"from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

# Try to load environment variables
try:
    from dotenv import load_dotenv
    ROOT_DIR = Path(__file__).parent
    load_dotenv(ROOT_DIR / '.env')
except ImportError:
    pass

# MongoDB connection with fallback
try:
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'pharmgpt')]
    print(f\"Connected to MongoDB at: {mongo_url}\")
except Exception as e:
    print(f\"MongoDB connection error: {e}\")
    client = None
    db = None

# Create the main app
app = FastAPI(title=\"PharmaGPT API\", description=\"AI-powered pharmaceutical quality analysis\")

# Create a router with the /api prefix
api_router = APIRouter(prefix=\"/api\")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[\"*\"],
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

# Basic models
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str  # 'user' or 'assistant'
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    regulatory_context: Optional[str] = \"GMP\"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    suggestions: Optional[List[str]] = None

class Consultant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    title: str
    expertise_areas: List[str]
    experience_years: int
    bio: str
    hourly_rate: float
    rating: float = 0.0
    total_consultations: int = 0
    is_active: bool = True

# Image Analysis Types
IMAGE_ANALYSIS_TYPES = {
    \"laboratory_results\": {
        \"name\": \"Laboratory Results Analysis\",
        \"description\": \"Analyze test results, chromatograms, spectra, and analytical data\",
        \"focus_areas\": [\"accuracy\", \"precision\", \"method validation\", \"out-of-specification results\"]
    },
    \"product_quality\": {
        \"name\": \"Product Quality Assessment\", 
        \"description\": \"Evaluate tablet/capsule appearance, packaging, and physical defects\",
        \"focus_areas\": [\"appearance\", \"uniformity\", \"contamination\", \"packaging integrity\"]
    },
    \"equipment_readings\": {
        \"name\": \"Equipment & Instrument Analysis\",
        \"description\": \"Assess equipment displays, calibration records, and monitoring data\",
        \"focus_areas\": [\"calibration status\", \"alarm conditions\", \"trend analysis\", \"maintenance needs\"]
    },
    \"process_monitoring\": {
        \"name\": \"Process Monitoring\",
        \"description\": \"Review process control charts, batch records, and manufacturing data\",
        \"focus_areas\": [\"process control\", \"trend analysis\", \"deviation identification\", \"statistical analysis\"]
    },
    \"document_analysis\": {
        \"name\": \"Document Analysis\",
        \"description\": \"Analyze regulatory documents, procedures, and compliance records\",
        \"focus_areas\": [\"compliance review\", \"document completeness\", \"regulatory requirements\", \"format validation\"]
    },
    \"contamination_assessment\": {
        \"name\": \"Contamination Assessment\",
        \"description\": \"Identify and assess microbial, particulate, or chemical contamination\",
        \"focus_areas\": [\"contamination identification\", \"source investigation\", \"risk assessment\", \"remediation recommendations\"]
    }
}

# Basic chat endpoint (without AI for now)
@api_router.post(\"/chat\", response_model=ChatResponse)
async def chat_basic(request: ChatRequest):
    \"\"\"Basic chat endpoint - returns helpful pharmaceutical guidance\"\"\"
    try:
        # Store user message if database available
        if db:
            user_message_doc = ChatMessage(
                session_id=request.session_id,
                message=request.message,
                sender=\"user\"
            ).dict()
            await db.chat_messages.insert_one(user_message_doc)
        
        # Generate basic pharmaceutical response
        response_text = f\"Thank you for your pharmaceutical question regarding: '{request.message}'. \"
        
        if \"batch failure\" in request.message.lower():
            response_text += \"For batch failures, I recommend following a systematic investigation approach: 1) Review batch records, 2) Check raw material specifications, 3) Verify equipment calibration, 4) Assess environmental conditions, 5) Perform root cause analysis using 5-Why or Fishbone methods.\"
        elif \"gmp\" in request.message.lower() or \"compliance\" in request.message.lower():
            response_text += f\"For {request.regulatory_context} compliance, ensure all processes follow current good manufacturing practices, maintain proper documentation, and implement robust quality systems.\"
        elif \"quality\" in request.message.lower():
            response_text += \"Quality assurance in pharmaceutical manufacturing requires systematic monitoring, proper documentation, risk assessment, and continuous improvement processes.\"
        else:
            response_text += \"I'm here to help with pharmaceutical quality, regulatory compliance, process troubleshooting, and manufacturing guidance. Please provide more specific details for targeted assistance.\"
        
        # Store AI response if database available
        if db:
            ai_message_doc = ChatMessage(
                session_id=request.session_id,
                message=response_text,
                sender=\"assistant\"
            ).dict()
            await db.chat_messages.insert_one(ai_message_doc)
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            suggestions=[
                \"Perform root cause analysis\",
                \"Review batch documentation\", 
                \"Check regulatory compliance\",
                \"Assess quality metrics\"
            ]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f\"Chat processing failed: {str(e)}\")

# Image Analysis endpoints
@api_router.get(\"/image-analysis/types\")
async def get_image_analysis_types():
    \"\"\"Get available image analysis types\"\"\"
    return IMAGE_ANALYSIS_TYPES

# Consultants endpoint
@api_router.get(\"/consultants\")
async def get_consultants():
    \"\"\"Get list of available consultants\"\"\"
    # Return sample consultant data
    sample_consultants = [
        {
            \"id\": \"consultant-1\",
            \"name\": \"Dr. Sarah Johnson\",
            \"title\": \"Senior Regulatory Affairs Consultant\", 
            \"expertise_areas\": [\"USFDA Submissions\", \"GMP Compliance\", \"Quality Systems\"],
            \"experience_years\": 15,
            \"bio\": \"Expert in pharmaceutical regulatory affairs with 15+ years experience\",
            \"hourly_rate\": 350.0,
            \"rating\": 4.9,
            \"total_consultations\": 127,
            \"is_active\": True
        },
        {
            \"id\": \"consultant-2\", 
            \"name\": \"Dr. Michael Chen\",
            \"title\": \"API Manufacturing Expert\",
            \"expertise_areas\": [\"API Development\", \"Process Optimization\", \"Green Chemistry\"],
            \"experience_years\": 12,
            \"bio\": \"Specialist in API manufacturing and process development\",
            \"hourly_rate\": 320.0,
            \"rating\": 4.8,
            \"total_consultations\": 89,
            \"is_active\": True
        }
    ]
    
    return sample_consultants

# Troubleshooting templates
@api_router.get(\"/troubleshooting-templates\")
async def get_troubleshooting_templates():
    \"\"\"Get troubleshooting templates\"\"\"
    return {
        \"batch_failure\": {
            \"name\": \"Batch Failure Investigation\",
            \"questions\": [
                \"What is the batch number and product?\",
                \"At which stage did the failure occur?\", 
                \"What were the observed deviations?\",
                \"Were there any equipment malfunctions?\",
                \"What were the environmental conditions?\"
            ],
            \"regulatory_focus\": [\"GMP\", \"ICH Q10\"]
        },
        \"process_optimization\": {
            \"name\": \"Process Optimization\", 
            \"questions\": [
                \"Which process step needs optimization?\",
                \"What are the current performance metrics?\",
                \"What are the target specifications?\",
                \"Are there any regulatory constraints?\"
            ],
            \"regulatory_focus\": [\"ICH Q8\", \"ICH Q9\", \"ICH Q10\"]
        },
        \"contamination_investigation\": {
            \"name\": \"Contamination Investigation\",
            \"questions\": [
                \"What type of contamination was detected?\",
                \"When was it first identified?\",
                \"Which products/batches are affected?\",
                \"What is the potential source?\"
            ],
            \"regulatory_focus\": [\"GMP\", \"USFDA\", \"ICH Q7\"]
        }
    }

# Chat history
@api_router.get(\"/chat-history/{session_id}\")
async def get_chat_history(session_id: str):
    \"\"\"Get chat history for a session\"\"\"
    if not db:
        return []
        
    try:
        messages = await db.chat_messages.find(
            {\"session_id\": session_id}
        ).sort(\"timestamp\", 1).to_list(1000)
        
        return [ChatMessage(**msg) for msg in messages]
    except Exception as e:
        return []

# Health check endpoint
@api_router.get(\"/\")
async def root():
    \"\"\"Health check endpoint\"\"\"
    return {
        \"message\": \"PharmaGPT API - Ready to assist with pharmaceutical quality and R&D\",
        \"status\": \"healthy\",
        \"database\": \"connected\" if db else \"not available\",
        \"features\": [
            \"Chat assistance\",
            \"Visual Quality Analysis types\", 
            \"Expert consultants\",
            \"Troubleshooting templates\"
        ]
    }

# Include the router in the main app
app.include_router(api_router)

# Startup event
@app.on_event(\"startup\")
async def startup_event():
    \"\"\"Application startup\"\"\"
    print(\"PharmaGPT API starting up...\")
    if db:
        print(\"Database connection established\")
    else:
        print(\"Running without database connection\")

if __name__ == \"__main__\":
    import uvicorn
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)"
