from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime

# Import AI functionality
try:
    import openai
    HAS_AI = True
    print("✅ OpenAI integration loaded successfully")
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
- Focus on practical solutions for pharmaceutical professionals"""

# Enhanced Chat endpoint with OpenAI
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        # Store user message
        if db:
            user_message_doc = ChatMessage(
                session_id=request.session_id,
                message=request.message,
                sender="user"
            ).dict()
            await db.chat_messages.insert_one(user_message_doc)
        
        # Get AI response if available
        if HAS_AI and os.environ.get('EMERGENT_LLM_KEY'):
            try:
                openai_client = openai.OpenAI(api_key=os.environ['EMERGENT_LLM_KEY'])
                
                system_message = get_pharma_system_message(request.regulatory_context or "GMP")
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": request.message}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                ai_response = response.choices[0].message.content
                
            except Exception as e:
                print(f"OpenAI Error: {e}")
                ai_response = generate_fallback_response(request.message, request.regulatory_context)
        else:
            ai_response = generate_fallback_response(request.message, request.regulatory_context)
        
        # Store AI response
        if db:
            ai_message_doc = ChatMessage(
                session_id=request.session_id,
                message=ai_response,
                sender="assistant"
            ).dict()
            await db.chat_messages.insert_one(ai_message_doc)
        
        # Generate suggestions
        suggestions = generate_suggestions(request.message)
        
        return ChatResponse(
            response=ai_response,
            session_id=request.session_id,
            suggestions=suggestions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def generate_fallback_response(message: str, regulatory_context: str = "GMP"):
    """Generate intelligent fallback responses for pharmaceutical queries"""
    message_lower = message.lower()
    
    if "batch failure" in message_lower or "batch failed" in message_lower:
        return f"""For batch failure investigation, I recommend following a systematic approach:

1. **Immediate Assessment**: Secure and quarantine the affected batch
2. **Documentation Review**: Examine batch records, raw material certificates, and process parameters
3. **Root Cause Analysis**: Use 5-Why or Fishbone methodology to identify underlying causes
4. **Equipment Verification**: Check calibration status and maintenance records
5. **Environmental Factors**: Review temperature, humidity, and facility conditions
6. **{regulatory_context} Compliance**: Ensure investigation follows regulatory requirements
7. **CAPA Implementation**: Develop corrective and preventive actions

Would you like me to elaborate on any specific aspect of the investigation?"""

    elif "gmp" in message_lower or "compliance" in message_lower:
        return f"""For {regulatory_context} compliance, key focus areas include:

**Documentation & Records**:
- Complete and accurate batch records
- Equipment calibration and maintenance logs
- Personnel training documentation
- Change control procedures

**Quality Systems**:
- Robust quality management system
- Risk assessment processes
- Deviation and CAPA management
- Supplier qualification and monitoring

**Process Controls**:
- Critical process parameter monitoring
- In-process testing and release procedures
- Environmental monitoring programs
- Cleaning validation

How can I assist you with specific compliance requirements?"""

    else:
        return f"""Thank you for your pharmaceutical question: "{message}"

As PharmaGPT, I'm specialized in:
- API manufacturing and process troubleshooting
- Quality assurance and {regulatory_context} compliance
- Regulatory documentation and submissions
- Investigation methodologies (5-Why, Fishbone, CAPA)
- Process optimization and validation

Please provide more specific details about your challenge, and I'll offer targeted pharmaceutical expertise and actionable recommendations."""

def generate_suggestions(message: str):
    """Generate contextual suggestions based on user message"""
    message_lower = message.lower()
    
    if "batch failure" in message_lower:
        return [
            "Perform 5-Why root cause analysis",
            "Review critical process parameters", 
            "Check equipment calibration records",
            "Assess environmental conditions"
        ]
    elif "gmp" in message_lower or "compliance" in message_lower:
        return [
            "Review current compliance status",
            "Update documentation procedures",
            "Schedule internal audit",
            "Check regulatory updates"
        ]
    else:
        return [
            "Ask about batch failures",
            "Inquire about GMP compliance", 
            "Request process guidance",
            "Get contamination help"
        ]

# Image Analysis Types
@app.get("/api/image-analysis/types")
def get_analysis_types():
    return {
        "laboratory_results": {
            "name": "Laboratory Results Analysis",
            "description": "Analyze test results, chromatograms, spectra, and analytical data",
            "focus_areas": ["accuracy", "precision", "method validation", "out-of-specification results"]
        },
        "product_quality": {
            "name": "Product Quality Assessment", 
            "description": "Evaluate tablet/capsule appearance, packaging, and physical defects",
            "focus_areas": ["appearance", "uniformity", "contamination", "packaging integrity"]
        },
        "equipment_readings": {
            "name": "Equipment & Instrument Analysis",
            "description": "Assess equipment displays, calibration records, and monitoring data",
            "focus_areas": ["calibration status", "alarm conditions", "trend analysis", "maintenance needs"]
        },
        "process_monitoring": {
            "name": "Process Monitoring",
            "description": "Review process control charts, batch records, and manufacturing data",
            "focus_areas": ["process control", "trend analysis", "deviation identification", "statistical analysis"]
        },
        "document_analysis": {
            "name": "Document Analysis", 
            "description": "Analyze regulatory documents, procedures, and compliance records",
            "focus_areas": ["compliance review", "document completeness", "regulatory requirements", "format validation"]
        },
        "contamination_assessment": {
            "name": "Contamination Assessment",
            "description": "Identify and assess microbial, particulate, or chemical contamination",
            "focus_areas": ["contamination identification", "source investigation", "risk assessment", "remediation recommendations"]
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
        "features": ["AI Chat", "Image Analysis Types", "Expert Consultants"],
        "version": "2.0.0-enhanced"
    }

@app.get("/api/")
def api_root():
    return {
        "message": "PharmaGPT API - Ready for pharmaceutical assistance", 
        "ai_enabled": HAS_AI
    }

@app.on_event("startup")
async def startup():
    print("🚀 PharmaGPT API Enhanced v2.0 starting...")
    print(f"🤖 AI Features: {'Enabled with OpenAI' if HAS_AI else 'Disabled'}")
    print(f"🗄️ Database: {'Connected' if db else 'Fallback mode'}")
