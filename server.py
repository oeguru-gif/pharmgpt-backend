from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

# Import AI functionality
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    HAS_AI = True
    print("✅ AI integration loaded successfully")
except ImportError:
    HAS_AI = False
    print("❌ AI integration not available")

# Load environment
try:
    from dotenv import load_dotenv
    ROOT_DIR = Path(__file__).parent
    load_dotenv(ROOT_DIR / '.env')
except ImportError:
    pass

# MongoDB connection
try:
    mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ.get('DB_NAME', 'pharmgpt')]
    print(f"✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB connection error: {e}")
    client = None
    db = None

app = FastAPI(title="PharmaGPT API", description="AI-powered pharmaceutical quality analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ChatRequest(BaseModel):
    message: str
    session_id: str
    regulatory_context: Optional[str] = "GMP"

class ChatResponse(BaseModel):
    response: str
    session_id: str
    suggestions: Optional[List[str]] = None

class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# AI System Message
def get_pharma_system_message(regulatory_context: str = "GMP"):
    return f"""You are PharmaGPT, an expert AI assistant specialized in pharmaceutical quality, R&D, and API manufacturing. You have deep knowledge of:

REGULATORY STANDARDS: {regulatory_context}, ISO standards, USFDA, KFDA, EDQM, and PMDA requirements
EXPERTISE AREAS:
- API (Active Pharmaceutical Ingredient) manufacturing processes
- Process troubleshooting and batch failure analysis
- Quality control and quality assurance
- Regulatory compliance and documentation
- Investigation methodologies (5-Why, Fishbone analysis, CAPA)
- Green chemistry and sustainable processes

RESPONSE GUIDELINES:
- Always consider {regulatory_context} compliance in your responses
- Provide specific, actionable recommendations
- Include relevant regulatory considerations
- Suggest investigation approaches when troubleshooting
- Reference appropriate industry standards and guidelines
- Focus on practical solutions for pharmaceutical professionals

When discussing process issues, always consider:
1. Root cause analysis approaches
2. Risk assessment implications
3. Regulatory reporting requirements
4. CAPA (Corrective and Preventive Actions) recommendations
5. Impact on product quality and patient safety"""

# Enhanced Chat endpoint with AI
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        # Store user message
        if db:
            user_message_doc = ChatMessage(
                session_id=request.session_id,
                message=request.message,
                sender="user",
                regulatory_context=request.regulatory_context
            ).dict()
            await db.chat_messages.insert_one(user_message_doc)
        
        # Get AI response if available
        if HAS_AI and os.environ.get('EMERGENT_LLM_KEY'):
            system_message = get_pharma_system_message(request.regulatory_context or "GMP")
            
            chat = LlmChat(
                api_key=os.environ['EMERGENT_LLM_KEY'],
                session_id=request.session_id,
                system_message=system_message
            ).with_model("openai", "gpt-4o-mini")
            
            user_msg = UserMessage(text=request.message)
            ai_response = await chat.send_message(user_msg)
        else:
            # Fallback response
            ai_response = f"Thank you for your pharmaceutical question: '{request.message}'. "
            if "batch failure" in request.message.lower():
                ai_response += "For batch failures, follow systematic investigation: 1) Review batch records, 2) Check raw materials, 3) Verify equipment calibration, 4) Assess environmental conditions, 5) Perform root cause analysis."
            elif "gmp" in request.message.lower():
                ai_response += f"For {request.regulatory_context} compliance, ensure proper documentation, quality systems, and continuous monitoring."
            else:
                ai_response += "I'm ready to help with pharmaceutical quality, regulatory compliance, and manufacturing guidance."
        
        # Store AI response
        if db:
            ai_message_doc = ChatMessage(
                session_id=request.session_id,
                message=ai_response,
                sender="assistant"
            ).dict()
            await db.chat_messages.insert_one(ai_message_doc)
        
        # Generate suggestions
        suggestions = []
        if "batch failure" in request.message.lower():
            suggestions = ["Perform 5-Why analysis", "Review critical parameters", "Check equipment calibration"]
        elif "regulatory" in request.message.lower():
            suggestions = ["Review compliance status", "Check recent updates", "Prepare documentation"]
        
        return ChatResponse(
            response=ai_response,
            session_id=request.session_id,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Image Analysis Types
@app.get("/api/image-analysis/types")
def get_analysis_types():
    return {
        "laboratory_results": {
            "name": "Laboratory Results Analysis",
            "description": "Analyze test results, chromatograms, spectra, and analytical data",
            "focus_areas": ["accuracy", "precision", "method validation"]
        },
        "product_quality": {
            "name": "Product Quality Assessment", 
            "description": "Evaluate tablet/capsule appearance, packaging, and defects",
            "focus_areas": ["appearance", "uniformity", "contamination"]
        },
        "equipment_readings": {
            "name": "Equipment Analysis",
            "description": "Assess equipment displays and monitoring data",
            "focus_areas": ["calibration", "trends", "maintenance"]
        },
        "process_monitoring": {
            "name": "Process Monitoring",
            "description": "Review control charts and manufacturing data",
            "focus_areas": ["control", "deviations", "analysis"]
        },
        "document_analysis": {
            "name": "Document Analysis", 
            "description": "Analyze regulatory documents and procedures",
            "focus_areas": ["compliance", "completeness", "validation"]
        },
        "contamination_assessment": {
            "name": "Contamination Assessment",
            "description": "Identify and assess contamination issues",
            "focus_areas": ["identification", "risk", "remediation"]
        }
    }

# Consultants
@app.get("/api/consultants")
def get_consultants():
    return [
        {
            "id": "consultant-1",
            "name": "Dr. Sarah Johnson",
            "title": "Senior Regulatory Affairs Consultant",
            "expertise_areas": ["USFDA Submissions", "GMP Compliance", "Quality Systems"],
            "experience_years": 15,
            "bio": "Expert in pharmaceutical regulatory affairs with 15+ years experience",
            "hourly_rate": 350.0,
            "rating": 4.9,
            "total_consultations": 127,
            "is_active": True
        },
        {
            "id": "consultant-2", 
            "name": "Dr. Michael Chen",
            "title": "API Manufacturing Expert",
            "expertise_areas": ["API Development", "Process Optimization"],
            "experience_years": 12,
            "bio": "Specialist in API manufacturing and process development",
            "hourly_rate": 320.0,
            "rating": 4.8,
            "total_consultations": 89,
            "is_active": True
        }
    ]

# Chat history
@app.get("/api/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    if not db:
        return []
    try:
        messages = await db.chat_messages.find(
            {"session_id": session_id}
        ).sort("timestamp", 1).to_list(1000)
        return [ChatMessage(**msg) for msg in messages]
    except:
        return []

# Health check
@app.get("/")
def root():
    return {
        "message": "PharmaGPT API - AI-powered pharmaceutical assistance",
        "status": "healthy",
        "ai_enabled": HAS_AI,
        "database": "connected" if db else "fallback",
        "features": ["AI Chat", "Image Analysis Types", "Expert Consultants"]
    }

@app.get("/api/")
def api_root():
    return {"message": "PharmaGPT API - Ready for pharmaceutical assistance", "ai_enabled": HAS_AI}

@app.on_event("startup")
async def startup():
    print("🚀 PharmaGPT API starting...")
    print(f"🤖 AI Features: {'Enabled' if HAS_AI else 'Disabled'}")
    print(f"🗄️ Database: {'Connected' if db else 'Fallback mode'}")
