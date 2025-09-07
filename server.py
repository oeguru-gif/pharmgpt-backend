**This is too big for one copy/paste! Let me give you the COMPLETE file content in one block:**
Action: cd /app/digitalocean-upload && echo "📄 COMPLETE server.py FILE CONTENT:" && echo "=== COPY EVERYTHING BELOW THIS LINE ===" && cat server.py
Observation: COMPLETE server.py FILE CONTENT:
=== COPY EVERYTHING BELOW THIS LINE ===
from fastapi import FastAPI, APIRouter, HTTPException, Request, File, UploadFile
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import base64
from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent
from emergentintegrations.payments.stripe.checkout import StripeCheckout, CheckoutSessionResponse, CheckoutStatusResponse, CheckoutSessionRequest

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Consultation Packages - FIXED PRICING (Security: Never accept amounts from frontend)
CONSULTATION_PACKAGES = {
    "quick_consultation": {
        "name": "Quick Consultation",
        "duration": 30,
        "price": 299.00,
        "description": "30-minute expert consultation for specific pharmaceutical questions",
        "features": ["Expert advice", "Email summary", "Follow-up questions"]
    },
    "standard_consultation": {
        "name": "Standard Consultation", 
        "duration": 60,
        "price": 499.00,
        "description": "1-hour comprehensive consultation with detailed analysis",
        "features": ["In-depth analysis", "Written report", "Action plan", "Follow-up support"]
    },
    "premium_consultation": {
        "name": "Premium Consultation",
        "duration": 90,
        "price": 799.00,
        "description": "90-minute comprehensive consultation with regulatory expert",
        "features": ["Regulatory review", "Compliance guidance", "Detailed documentation", "Extended support"]
    },
    "emergency_consultation": {
        "name": "Emergency Consultation",
        "duration": 45,
        "price": 899.00,
        "description": "Urgent consultation available within 2 hours",
        "features": ["Immediate response", "Emergency support", "Crisis management", "24/7 availability"]
    }
}

# Existing models (keeping all previous models)
class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    message: str
    sender: str  # 'user' or 'assistant'
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    regulatory_context: Optional[str] = None
    process_type: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image
    analysis_type: Optional[str] = None  # For image analysis

class ChatRequest(BaseModel):
    message: str
    session_id: str
    regulatory_context: Optional[str] = "GMP"
    process_type: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image
    analysis_type: Optional[str] = None  # For image analysis

class ChatResponse(BaseModel):
    response: str
    session_id: str
    suggestions: Optional[List[str]] = None

# New Image Analysis Models
class ImageAnalysisRequest(BaseModel):
    image_data: str  # Base64 encoded image
    analysis_type: str  # Type of analysis requested 
    description: Optional[str] = None  # Optional description from user
    regulatory_context: str = "GMP"
    user_id: str

class ImageAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    image_data: str  # Base64 encoded image
    analysis_type: str
    description: Optional[str] = None
    regulatory_context: str
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    quality_score: Optional[int] = None  # 1-10 quality score
    compliance_issues: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserProfile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    regulatory_preferences: List[str] = Field(default=["GMP", "ISO"])
    company_type: str = "API_Manufacturing"
    experience_level: str = "Intermediate"
    created_at: datetime = Field(default_factory=datetime.utcnow)

# New Consultant System Models
class Consultant(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    title: str
    expertise_areas: List[str]
    experience_years: int
    education: List[str]
    certifications: List[str]
    bio: str
    hourly_rate: float
    availability_schedule: Dict[str, List[str]]  # day -> available hours
    rating: float = 0.0
    total_consultations: int = 0
    languages: List[str] = Field(default=["English"])
    profile_image: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ConsultationRequest(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    consultant_id: str
    package_id: str
    title: str
    description: str
    preferred_date: datetime
    preferred_time: str
    urgency_level: str = "normal"  # normal, high, emergency
    regulatory_context: str = "GMP"
    company_name: Optional[str] = None
    contact_email: str
    phone_number: Optional[str] = None
    status: str = "pending"  # pending, confirmed, completed, cancelled
    payment_status: str = "pending"  # pending, paid, failed, refunded
    payment_session_id: Optional[str] = None
    scheduled_datetime: Optional[datetime] = None
    meeting_link: Optional[str] = None
    consultation_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConsultationBookingRequest(BaseModel):
    consultant_id: str
    package_id: str
    title: str
    description: str
    preferred_date: str  # YYYY-MM-DD format
    preferred_time: str  # HH:MM format
    urgency_level: str = "normal"
    regulatory_context: str = "GMP"
    company_name: Optional[str] = None
    contact_email: str
    phone_number: Optional[str] = None

class PaymentTransaction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: Optional[str] = None
    consultation_request_id: Optional[str] = None
    package_id: str
    amount: float
    currency: str = "usd"
    payment_status: str = "pending"  # pending, paid, failed, expired, cancelled
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ConsultationReview(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    consultation_request_id: str
    consultant_id: str
    user_id: str
    rating: int  # 1-5 stars
    review_text: str
    would_recommend: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Investigation Tools Models (keeping existing)
class FiveWhyAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    problem_statement: str
    whys: List[str] = Field(default_factory=list)
    root_cause: Optional[str] = None
    recommendations: List[str] = Field(default_factory=list)
    regulatory_context: str = "GMP"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FiveWhyRequest(BaseModel):
    problem_statement: str
    user_id: str
    regulatory_context: Optional[str] = "GMP"

class FishboneAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    problem_statement: str
    categories: Dict[str, List[str]] = Field(default_factory=dict)
    root_causes: List[str] = Field(default_factory=list)
    regulatory_context: str = "GMP"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FishboneRequest(BaseModel):
    problem_statement: str
    user_id: str
    regulatory_context: Optional[str] = "GMP"

class CAPAAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    problem_statement: str
    root_causes: List[str]
    corrective_actions: List[str] = Field(default_factory=list)
    preventive_actions: List[str] = Field(default_factory=list)
    regulatory_requirements: List[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    responsible_parties: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CAPARequest(BaseModel):
    problem_statement: str
    root_causes: List[str]
    user_id: str
    regulatory_context: Optional[str] = "GMP"

# Green Chemistry Models (keeping existing)
class GreenChemistryAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    process_description: str
    current_solvents: List[str] = Field(default_factory=list)
    e_factor: Optional[float] = None
    sustainability_score: Optional[int] = None
    recommendations: List[str] = Field(default_factory=list)
    alternative_solvents: List[str] = Field(default_factory=list)
    green_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GreenChemistryRequest(BaseModel):
    process_description: str
    current_solvents: List[str]
    user_id: str
    target_metrics: Optional[List[str]] = None

# Enhanced proofreading detection keywords
PROOFREADING_KEYWORDS = [
    "misleading information", "regulatory submission", "inconsistencies", "errors",
    "inconsistent language", "regulatory requirements", "references", "efficacy",
    "safety claims", "labeling information", "clinical trial data", "regulatory guidelines",
    "regulatory updates", "risk management", "document review", "compliance",
    "submission", "proofread", "review this", "check this", "analyze this document"
]

# Image Analysis Types
IMAGE_ANALYSIS_TYPES = {
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

# Initialize Stripe Checkout
def get_stripe_checkout(host_url: str) -> StripeCheckout:
    webhook_url = f"{host_url}/api/webhook/stripe"
    return StripeCheckout(api_key=os.environ['STRIPE_API_KEY'], webhook_url=webhook_url)

# Initialize LLM Chat with pharma-specific system message
def get_pharma_system_message(regulatory_context: str = "GMP", process_type: str = None, specialty: str = None):
    base_message = f"""You are PharmaGPT, an expert AI assistant specialized in pharmaceutical quality, R&D, and API manufacturing. You have deep knowledge of:

REGULATORY STANDARDS: {regulatory_context}, ISO standards, USFDA, KFDA, EDQM, and PMDA requirements
EXPERTISE AREAS:
- API (Active Pharmaceutical Ingredient) manufacturing processes
- Process troubleshooting and batch failure analysis
- Quality control and quality assurance
- Regulatory compliance and documentation
- Investigation methodologies (5-Why, Fishbone analysis, CAPA)
- Green chemistry and sustainable processes
- Regulatory document proofreading and review
- Visual quality analysis and image interpretation

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
5. Impact on product quality and patient safety
"""
    
    if process_type:
        if process_type == "regulatory_proofreading":
            base_message += f"""

SPECIALTY MODE: Regulatory Document Proofreading Expert
- Focus on pharmaceutical regulatory submissions, compliance documentation, and quality manuals
- Identify misleading information, inconsistencies, and potential compliance issues
- Suggest improvements for clarity, accuracy, and regulatory compliance
- Ensure alignment with {regulatory_context} and current regulatory guidelines
- Provide comprehensive document review with specific recommendations
- Check for consistent terminology, proper references, and complete coverage of requirements"""
        elif process_type == "image_analysis":
            base_message += f"""

SPECIALTY MODE: Visual Quality Analysis Expert
- Analyze pharmaceutical images including laboratory results, product quality, equipment readings
- Identify quality issues, deviations, and compliance concerns from visual data
- Provide detailed assessment of pharmaceutical processes and products shown in images
- Focus on {regulatory_context} compliance and pharmaceutical industry standards
- Generate quality scores and specific recommendations based on visual analysis
- Identify potential contamination, defects, or process deviations
- Suggest corrective actions and investigation approaches based on visual findings"""
        else:
            base_message += f"\n\nCURRENT FOCUS: {process_type} process troubleshooting and optimization."
    
    if specialty == "investigation":
        base_message += f"\n\nSPECIALTY MODE: Investigation Tools - Focus on systematic root cause analysis, 5-Why methodology, Fishbone diagrams, and CAPA development following {regulatory_context} guidelines."
    
    elif specialty == "green_chemistry":
        base_message += f"\n\nSPECIALTY MODE: Green Chemistry Advisor - Focus on sustainable pharmaceutical processes, solvent reduction, E-factor optimization, green metrics, and environmentally friendly alternatives while maintaining {regulatory_context} compliance."
    
    return base_message

def detect_proofreading_request(message: str) -> bool:
    """Detect if the message is requesting document proofreading or review"""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in PROOFREADING_KEYWORDS)

def get_image_analysis_prompt(analysis_type: str, regulatory_context: str, user_description: str = None) -> str:
    """Generate specialized image analysis prompt based on type"""
    
    analysis_config = IMAGE_ANALYSIS_TYPES.get(analysis_type, IMAGE_ANALYSIS_TYPES["laboratory_results"])
    
    base_prompt = f"""Analyze this pharmaceutical image as a {analysis_config['name']}.

ANALYSIS TYPE: {analysis_config['description']}
REGULATORY CONTEXT: {regulatory_context}
FOCUS AREAS: {', '.join(analysis_config['focus_areas'])}

"""
    
    if user_description:
        base_prompt += f"USER CONTEXT: {user_description}\n\n"
    
    if analysis_type == "laboratory_results":
        base_prompt += """Please provide detailed analysis including:
1. RESULTS INTERPRETATION: What do the values/patterns indicate?
2. QUALITY ASSESSMENT: Are results within acceptable limits?
3. METHOD VALIDATION: Any concerns with analytical method?
4. REGULATORY COMPLIANCE: Alignment with pharmacopeial standards
5. RECOMMENDATIONS: Next steps if deviations are found
6. QUALITY SCORE: Rate overall quality 1-10 (10 being excellent)
"""
    
    elif analysis_type == "product_quality":
        base_prompt += """Please provide detailed analysis including:
1. APPEARANCE ASSESSMENT: Color, shape, surface defects
2. UNIFORMITY EVALUATION: Size, weight, coating consistency
3. CONTAMINATION CHECK: Foreign particles, contamination signs
4. PACKAGING INTEGRITY: Container closure, labeling accuracy
5. COMPLIANCE ISSUES: GMP deviations or quality concerns
6. QUALITY SCORE: Rate overall product quality 1-10
"""
    
    elif analysis_type == "equipment_readings":
        base_prompt += """Please provide detailed analysis including:
1. READING INTERPRETATION: Current values and normal ranges
2. CALIBRATION STATUS: Is equipment properly calibrated?
3. ALARM CONDITIONS: Any warnings or critical alerts?
4. TREND ANALYSIS: Historical data patterns if visible
5. MAINTENANCE NEEDS: Preventive or corrective actions required
6. QUALITY SCORE: Rate equipment performance 1-10
"""
    
    elif analysis_type == "process_monitoring":
        base_prompt += """Please provide detailed analysis including:
1. PROCESS CONTROL: Are parameters within control limits?
2. STATISTICAL ANALYSIS: Trends, patterns, control chart interpretation
3. DEVIATION IDENTIFICATION: Out-of-specification conditions
4. BATCH RECORD REVIEW: Completeness and accuracy assessment
5. PROCESS IMPROVEMENT: Optimization recommendations
6. QUALITY SCORE: Rate process control 1-10
"""
    
    elif analysis_type == "document_analysis":
        base_prompt += """Please provide detailed analysis including:
1. COMPLIANCE REVIEW: Regulatory requirements adherence
2. DOCUMENT COMPLETENESS: Missing sections or information
3. FORMAT VALIDATION: Proper structure and organization
4. REGULATORY ALIGNMENT: Consistency with guidelines
5. IMPROVEMENT SUGGESTIONS: Areas needing enhancement
6. QUALITY SCORE: Rate document quality 1-10
"""
    
    elif analysis_type == "contamination_assessment":
        base_prompt += """Please provide detailed analysis including:
1. CONTAMINATION IDENTIFICATION: Type and extent of contamination
2. SOURCE INVESTIGATION: Potential contamination sources
3. RISK ASSESSMENT: Impact on product quality and safety
4. IMMEDIATE ACTIONS: Urgent containment measures
5. REMEDIATION PLAN: Long-term corrective actions
6. QUALITY SCORE: Rate contamination severity 1-10 (1 being severe)
"""
    
    base_prompt += f"""

Format your response with clear sections and provide specific, actionable recommendations.
Consider {regulatory_context} requirements throughout your analysis.
"""
    
    return base_prompt

# Image Analysis Endpoints
@api_router.post("/image-analysis", response_model=ImageAnalysisResult)
async def analyze_image(request: ImageAnalysisRequest):
    """Analyze pharmaceutical images for quality assessment"""
    try:
        # Validate analysis type
        if request.analysis_type not in IMAGE_ANALYSIS_TYPES:
            raise HTTPException(status_code=400, detail="Invalid analysis type")
        
        # Generate specialized prompt
        analysis_prompt = get_image_analysis_prompt(
            request.analysis_type,
            request.regulatory_context,
            request.description
        )
        
        # Initialize AI with image analysis specialization
        system_message = get_pharma_system_message(
            request.regulatory_context,
            process_type="image_analysis"
        )
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"image_analysis_{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o")  # Use GPT-4 Vision for image analysis
        
        # Send image and prompt for analysis
        # Extract base64 data from data URL if present
        image_data = request.image_data
        if image_data.startswith('data:image/'):
            # Remove data URL prefix to get pure base64
            image_data = image_data.split(',')[1]
        
        image_content = ImageContent(image_base64=image_data)
        user_msg = UserMessage(text=analysis_prompt, file_contents=[image_content])
        ai_response = await chat.send_message(user_msg)
        
        # Parse AI response to extract structured data
        analysis_results = {}
        recommendations = []
        compliance_issues = []
        quality_score = None
        
        # Extract structured information from AI response
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'QUALITY SCORE:' in line.upper() or 'SCORE:' in line.upper():
                try:
                    # Extract numeric score
                    score_part = line.split(':')[1].strip()
                    quality_score = int(score_part.split()[0])
                except:
                    quality_score = 5  # Default score
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '•', '-')) and 'RECOMMENDATION' in line.upper():
                recommendations.append(line)
            elif 'COMPLIANCE' in line.upper() and ('ISSUE' in line.upper() or 'CONCERN' in line.upper()):
                compliance_issues.append(line)
            elif ':' in line and len(line) > 20:
                # Store key findings
                key, value = line.split(':', 1)
                analysis_results[key.strip()] = value.strip()
        
        # If no recommendations extracted, add some based on analysis type
        if not recommendations:
            if quality_score and quality_score < 7:
                recommendations = [
                    "Review current procedures and protocols",
                    "Investigate root causes of quality issues",
                    "Implement corrective actions as needed",
                    "Consider additional training or process improvements"
                ]
        
        # Create analysis record
        result = ImageAnalysisResult(
            user_id=request.user_id,
            image_data=request.image_data,
            analysis_type=request.analysis_type,
            description=request.description,
            regulatory_context=request.regulatory_context,
            analysis_results={"ai_analysis": ai_response, **analysis_results},
            recommendations=recommendations,
            quality_score=quality_score,
            compliance_issues=compliance_issues
        )
        
        # Store in database
        await db.image_analyses.insert_one(result.dict())
        
        return result
        
    except Exception as e:
        logging.error(f"Image analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@api_router.get("/image-analysis/types")
async def get_image_analysis_types():
    """Get available image analysis types"""
    return IMAGE_ANALYSIS_TYPES

@api_router.get("/image-analysis/history/{user_id}")
async def get_image_analysis_history(user_id: str):
    """Get user's image analysis history"""
    try:
        analyses = await db.image_analyses.find({"user_id": user_id}).sort("created_at", -1).to_list(50)
        return [ImageAnalysisResult(**analysis) for analysis in analyses]
    except Exception as e:
        logging.error(f"Image analysis history error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve image analysis history")

# Enhanced Chat endpoint with image support
@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        # Detect if this is a proofreading request
        is_proofreading = detect_proofreading_request(request.message)
        is_image_analysis = bool(request.image_data)
        
        process_type = None
        if is_proofreading:
            process_type = "regulatory_proofreading"
        elif is_image_analysis:
            process_type = "image_analysis"
        else:
            process_type = request.process_type
        
        # Store user message
        user_message_doc = ChatMessage(
            session_id=request.session_id,
            message=request.message,
            sender="user",
            regulatory_context=request.regulatory_context,
            process_type=process_type,
            image_data=request.image_data,
            analysis_type=request.analysis_type
        ).dict()
        
        await db.chat_messages.insert_one(user_message_doc)
        
        # Get AI response using Emergent LLM
        system_message = get_pharma_system_message(
            request.regulatory_context or "GMP",
            process_type
        )
        
        # Use GPT-4 Vision if image is provided, otherwise use GPT-4o-mini
        model_name = "gpt-4o" if is_image_analysis else "gpt-4o-mini"
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=request.session_id,
            system_message=system_message
        ).with_model("openai", model_name)
        
        # Send message (with or without image)
        if is_image_analysis and request.image_data:
            # Enhanced prompt for image analysis in chat
            enhanced_message = f"{request.message}\n\nPlease analyze this pharmaceutical image and provide quality assessment with specific recommendations."
            # Extract base64 data from data URL if present
            image_data = request.image_data
            if image_data.startswith('data:image/'):
                # Remove data URL prefix to get pure base64
                image_data = image_data.split(',')[1]
            
            image_content = ImageContent(image_base64=image_data)
            user_msg = UserMessage(text=enhanced_message, file_contents=[image_content])
            ai_response = await chat.send_message(user_msg)
        else:
            user_msg = UserMessage(text=request.message)
            ai_response = await chat.send_message(user_msg)
        
        # Store AI response
        ai_message_doc = ChatMessage(
            session_id=request.session_id,
            message=ai_response,
            sender="assistant",
            regulatory_context=request.regulatory_context,
            process_type=process_type,
            analysis_type=request.analysis_type
        ).dict()
        
        await db.chat_messages.insert_one(ai_message_doc)
        
        # Generate helpful suggestions based on context
        suggestions = []
        if is_image_analysis:
            suggestions = [
                "Request detailed quality analysis",
                "Identify compliance issues",
                "Get improvement recommendations", 
                "Analyze for regulatory requirements"
            ]
        elif is_proofreading or "document" in request.message.lower():
            suggestions = [
                "Check for regulatory compliance issues",
                "Review terminology consistency", 
                "Verify all required references",
                "Assess clarity and completeness"
            ]
        elif "batch failure" in request.message.lower() or "batch failed" in request.message.lower():
            suggestions = [
                "Perform 5-Why root cause analysis",
                "Review critical process parameters",
                "Check raw material specifications",
                "Investigate environmental conditions"
            ]
        elif "troubleshoot" in request.message.lower():
            suggestions = [
                "Review process flow diagram",
                "Check equipment calibration status",
                "Analyze trend data",
                "Perform risk assessment"
            ]
        elif "regulatory" in request.message.lower():
            suggestions = [
                "Review current compliance status",
                "Check recent regulatory updates",
                "Prepare documentation package",
                "Schedule internal audit"
            ]
        
        return ChatResponse(
            response=ai_response,
            session_id=request.session_id,
            suggestions=suggestions
        )
        
    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Consultation System Endpoints (keeping all existing)
@api_router.get("/consultants")
async def get_consultants():
    """Get list of available consultants"""
    try:
        consultants = await db.consultants.find({"is_active": True}).to_list(50)
        return [Consultant(**consultant) for consultant in consultants]
    except Exception as e:
        logging.error(f"Get consultants error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve consultants")

@api_router.get("/consultants/{consultant_id}")
async def get_consultant(consultant_id: str):
    """Get detailed consultant information"""
    try:
        consultant = await db.consultants.find_one({"id": consultant_id, "is_active": True})
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        return Consultant(**consultant)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get consultant error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve consultant")

@api_router.get("/consultation-packages")
async def get_consultation_packages():
    """Get available consultation packages with pricing"""
    return CONSULTATION_PACKAGES

@api_router.post("/consultations/book")
async def book_consultation(request: ConsultationBookingRequest, http_request: Request):
    """Book a consultation with payment processing"""
    try:
        # Validate consultant exists and is active
        consultant = await db.consultants.find_one({"id": request.consultant_id, "is_active": True})
        if not consultant:
            raise HTTPException(status_code=404, detail="Consultant not found")
        
        # Validate package exists
        if request.package_id not in CONSULTATION_PACKAGES:
            raise HTTPException(status_code=400, detail="Invalid consultation package")
        
        # Generate user_id if not provided (for non-authenticated users)
        user_id = f"guest_{uuid.uuid4().hex[:12]}"
        
        # Create consultation request
        consultation_request = ConsultationRequest(
            user_id=user_id,
            consultant_id=request.consultant_id,
            package_id=request.package_id,
            title=request.title,
            description=request.description,
            preferred_date=datetime.strptime(f"{request.preferred_date} {request.preferred_time}", "%Y-%m-%d %H:%M"),
            preferred_time=request.preferred_time,
            urgency_level=request.urgency_level,
            regulatory_context=request.regulatory_context,
            company_name=request.company_name,
            contact_email=request.contact_email,
            phone_number=request.phone_number
        )
        
        # Store consultation request
        await db.consultation_requests.insert_one(consultation_request.dict())
        
        # Get package details for payment
        package = CONSULTATION_PACKAGES[request.package_id]
        amount = package["price"]
        
        # Initialize Stripe checkout
        host_url = str(http_request.base_url).rstrip('/')
        stripe_checkout = get_stripe_checkout(host_url)
        
        # Create checkout session
        success_url = f"{host_url.replace('/api', '')}/consultation-success?session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{host_url.replace('/api', '')}/consultations"
        
        checkout_request = CheckoutSessionRequest(
            amount=amount,
            currency="usd",
            success_url=success_url,
            cancel_url=cancel_url,
            metadata={
                "consultation_request_id": consultation_request.id,
                "package_id": request.package_id,
                "user_id": user_id,
                "consultant_id": request.consultant_id,
                "type": "consultation_booking"
            }
        )
        
        session = await stripe_checkout.create_checkout_session(checkout_request)
        
        # Create payment transaction record
        payment_transaction = PaymentTransaction(
            session_id=session.session_id,
            user_id=user_id,
            consultation_request_id=consultation_request.id,
            package_id=request.package_id,
            amount=amount,
            currency="usd",
            payment_status="pending",
            metadata=checkout_request.metadata
        )
        
        await db.payment_transactions.insert_one(payment_transaction.dict())
        
        # Update consultation request with payment session ID
        await db.consultation_requests.update_one(
            {"id": consultation_request.id},
            {"$set": {"payment_session_id": session.session_id, "updated_at": datetime.utcnow()}}
        )
        
        return {
            "consultation_request_id": consultation_request.id,
            "checkout_url": session.url,
            "session_id": session.session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Book consultation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to book consultation")

@api_router.get("/payments/status/{session_id}")
async def get_payment_status(session_id: str, http_request: Request):
    """Get payment status for a checkout session"""
    try:
        # Initialize Stripe checkout
        host_url = str(http_request.base_url).rstrip('/')
        stripe_checkout = get_stripe_checkout(host_url)
        
        # Get checkout status from Stripe
        checkout_status = await stripe_checkout.get_checkout_status(session_id)
        
        # Find payment transaction
        payment_transaction = await db.payment_transactions.find_one({"session_id": session_id})
        if not payment_transaction:
            raise HTTPException(status_code=404, detail="Payment transaction not found")
        
        # Update payment status if changed
        if payment_transaction["payment_status"] != checkout_status.payment_status:
            await db.payment_transactions.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "payment_status": checkout_status.payment_status,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # If payment is successful, update consultation request
            if checkout_status.payment_status == "paid" and payment_transaction.get("consultation_request_id"):
                await db.consultation_requests.update_one(
                    {"id": payment_transaction["consultation_request_id"]},
                    {
                        "$set": {
                            "payment_status": "paid",
                            "status": "confirmed",
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
        
        return {
            "session_id": session_id,
            "payment_status": checkout_status.payment_status,
            "status": checkout_status.status,
            "amount_total": checkout_status.amount_total,
            "currency": checkout_status.currency,
            "consultation_request_id": payment_transaction.get("consultation_request_id")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get payment status error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get payment status")

@api_router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """Handle Stripe webhooks"""
    try:
        body = await request.body()
        signature = request.headers.get("Stripe-Signature")
        
        # Initialize Stripe checkout
        host_url = str(request.base_url).rstrip('/')
        stripe_checkout = get_stripe_checkout(host_url)
        
        # Handle webhook
        webhook_response = await stripe_checkout.handle_webhook(body, signature)
        
        # Update payment transaction based on webhook
        if webhook_response.session_id:
            await db.payment_transactions.update_one(
                {"session_id": webhook_response.session_id},
                {
                    "$set": {
                        "payment_status": webhook_response.payment_status,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            # Update consultation request if payment is successful
            if webhook_response.payment_status == "paid":
                payment_transaction = await db.payment_transactions.find_one({"session_id": webhook_response.session_id})
                if payment_transaction and payment_transaction.get("consultation_request_id"):
                    await db.consultation_requests.update_one(
                        {"id": payment_transaction["consultation_request_id"]},
                        {
                            "$set": {
                                "payment_status": "paid",
                                "status": "confirmed",
                                "updated_at": datetime.utcnow()
                            }
                        }
                    )
        
        return {"status": "success"}
        
    except Exception as e:
        logging.error(f"Stripe webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@api_router.get("/consultations/{consultation_id}")
async def get_consultation(consultation_id: str):
    """Get consultation details"""
    try:
        consultation = await db.consultation_requests.find_one({"id": consultation_id})
        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found")
        return ConsultationRequest(**consultation)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Get consultation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve consultation")

@api_router.get("/consultations/user/{user_id}")
async def get_user_consultations(user_id: str):
    """Get user's consultation history"""
    try:
        consultations = await db.consultation_requests.find({"user_id": user_id}).sort("created_at", -1).to_list(50)
        return [ConsultationRequest(**consultation) for consultation in consultations]
    except Exception as e:
        logging.error(f"Get user consultations error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve consultations")

@api_router.post("/consultations/{consultation_id}/review")
async def submit_consultation_review(consultation_id: str, review: ConsultationReview):
    """Submit a review for a completed consultation"""
    try:
        # Verify consultation exists and is completed
        consultation = await db.consultation_requests.find_one({"id": consultation_id, "status": "completed"})
        if not consultation:
            raise HTTPException(status_code=404, detail="Consultation not found or not completed")
        
        # Store review
        review.consultation_request_id = consultation_id
        review.consultant_id = consultation["consultant_id"]
        await db.consultation_reviews.insert_one(review.dict())
        
        # Update consultant rating
        reviews = await db.consultation_reviews.find({"consultant_id": consultation["consultant_id"]}).to_list(1000)
        if reviews:
            avg_rating = sum(review["rating"] for review in reviews) / len(reviews)
            await db.consultants.update_one(
                {"id": consultation["consultant_id"]},
                {"$set": {"rating": round(avg_rating, 1), "total_consultations": len(reviews)}}
            )
        
        return {"message": "Review submitted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Submit review error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to submit review")

# Investigation Tools Endpoints (keeping existing)
@api_router.post("/investigation/five-why", response_model=FiveWhyAnalysis)
async def generate_five_why_analysis(request: FiveWhyRequest):
    try:
        # Use AI to generate 5-Why analysis
        system_message = get_pharma_system_message(
            request.regulatory_context,
            specialty="investigation"
        )
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"five_why_{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
        
        prompt = f"""Perform a systematic 5-Why root cause analysis for this pharmaceutical problem:

PROBLEM: {request.problem_statement}
REGULATORY CONTEXT: {request.regulatory_context}

Please provide:
1. Five progressive "Why" questions and answers
2. The final root cause identified
3. Specific recommendations for corrective actions
4. Regulatory considerations for {request.regulatory_context}

Format your response as:
WHY 1: [question] - [answer]
WHY 2: [question] - [answer]
WHY 3: [question] - [answer]
WHY 4: [question] - [answer]
WHY 5: [question] - [answer]
ROOT CAUSE: [final root cause]
RECOMMENDATIONS: [numbered list of recommendations]"""
        
        user_msg = UserMessage(text=prompt)
        ai_response = await chat.send_message(user_msg)
        
        # Parse AI response to extract structured data
        whys = []
        root_cause = None
        recommendations = []
        
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('WHY '):
                whys.append(line)
            elif line.startswith('ROOT CAUSE:'):
                root_cause = line.replace('ROOT CAUSE:', '').strip()
            elif line.startswith(('RECOMMENDATIONS:', '1.', '2.', '3.', '4.', '5.', '-', '•')):
                if not line.startswith('RECOMMENDATIONS:'):
                    recommendations.append(line)
        
        # Create analysis record
        analysis = FiveWhyAnalysis(
            user_id=request.user_id,
            problem_statement=request.problem_statement,
            whys=whys,
            root_cause=root_cause,
            recommendations=recommendations,
            regulatory_context=request.regulatory_context
        )
        
        # Store in database
        await db.five_why_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        logging.error(f"5-Why analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"5-Why analysis failed: {str(e)}")

@api_router.post("/investigation/fishbone", response_model=FishboneAnalysis)
async def generate_fishbone_analysis(request: FishboneRequest):
    try:
        # Use AI to generate Fishbone analysis
        system_message = get_pharma_system_message(
            request.regulatory_context,
            specialty="investigation"
        )
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"fishbone_{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
        
        prompt = f"""Create a Fishbone (Ishikawa) diagram analysis for this pharmaceutical problem:

PROBLEM: {request.problem_statement}
REGULATORY CONTEXT: {request.regulatory_context}

Please organize potential causes into these pharmaceutical-specific categories:
1. MATERIALS (Raw materials, reagents, solvents)
2. METHODS (Procedures, processes, protocols)
3. MACHINES (Equipment, instruments, utilities)
4. MANPOWER (Personnel, training, competency)
5. MEASUREMENT (Testing, calibration, specifications)
6. ENVIRONMENT (Temperature, humidity, contamination)

For each category, provide 3-5 specific potential causes.
Then identify the most likely root causes based on pharmaceutical industry experience.

Format your response as:
MATERIALS:
- [cause 1]
- [cause 2]
...

METHODS:
- [cause 1]
- [cause 2]
...

[Continue for all categories]

MOST LIKELY ROOT CAUSES:
1. [primary root cause]
2. [secondary root cause]
3. [tertiary root cause]"""
        
        user_msg = UserMessage(text=prompt)
        ai_response = await chat.send_message(user_msg)
        
        # Parse AI response to extract structured data
        categories = {
            "Materials": [],
            "Methods": [],
            "Machines": [],
            "Manpower": [],
            "Measurement": [],
            "Environment": []
        }
        
        root_causes = []
        current_category = None
        
        lines = ai_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('MATERIALS:', 'METHODS:', 'MACHINES:', 'MANPOWER:', 'MEASUREMENT:', 'ENVIRONMENT:')):
                current_category = line.replace(':', '').title()
            elif line.startswith('- ') and current_category:
                categories[current_category].append(line[2:])
            elif line.startswith('MOST LIKELY ROOT CAUSES:'):
                current_category = None
            elif line.startswith(('1.', '2.', '3.')) and 'ROOT CAUSES' in ai_response.upper():
                root_causes.append(line)
        
        # Create analysis record
        analysis = FishboneAnalysis(
            user_id=request.user_id,
            problem_statement=request.problem_statement,
            categories=categories,
            root_causes=root_causes,
            regulatory_context=request.regulatory_context
        )
        
        # Store in database
        await db.fishbone_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        logging.error(f"Fishbone analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fishbone analysis failed: {str(e)}")

@api_router.post("/investigation/capa", response_model=CAPAAnalysis)
async def generate_capa_analysis(request: CAPARequest):
    try:
        # Use AI to generate CAPA analysis
        system_message = get_pharma_system_message(
            request.regulatory_context,
            specialty="investigation"
        )
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"capa_{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
        
        root_causes_text = "\n".join([f"- {cause}" for cause in request.root_causes])
        
        prompt = f"""Generate a comprehensive CAPA (Corrective and Preventive Actions) plan for this pharmaceutical issue:

PROBLEM: {request.problem_statement}
ROOT CAUSES IDENTIFIED:
{root_causes_text}

REGULATORY CONTEXT: {request.regulatory_context}

Please provide:
1. CORRECTIVE ACTIONS (immediate fixes to address the problem)
2. PREVENTIVE ACTIONS (long-term measures to prevent recurrence)
3. REGULATORY REQUIREMENTS specific to {request.regulatory_context}
4. IMPLEMENTATION TIMELINE (realistic timeframes)
5. RESPONSIBLE PARTIES (roles/departments)

Consider pharmaceutical industry best practices and {request.regulatory_context} compliance requirements.

Format your response as:
CORRECTIVE ACTIONS:
1. [action 1]
2. [action 2]
...

PREVENTIVE ACTIONS:
1. [action 1]
2. [action 2]
...

REGULATORY REQUIREMENTS:
- [requirement 1]
- [requirement 2]
...

TIMELINE: [overall timeline description]

RESPONSIBLE PARTIES:
- [party 1]: [responsibility]
- [party 2]: [responsibility]
..."""
        
        user_msg = UserMessage(text=prompt)
        ai_response = await chat.send_message(user_msg)
        
        # Parse AI response to extract structured data
        corrective_actions = []
        preventive_actions = []
        regulatory_requirements = []
        responsible_parties = []
        timeline = None
        
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('CORRECTIVE ACTIONS:'):
                current_section = 'corrective'
            elif line.startswith('PREVENTIVE ACTIONS:'):
                current_section = 'preventive'
            elif line.startswith('REGULATORY REQUIREMENTS:'):
                current_section = 'regulatory'
            elif line.startswith('TIMELINE:'):
                current_section = 'timeline'
                timeline = line.replace('TIMELINE:', '').strip()
            elif line.startswith('RESPONSIBLE PARTIES:'):
                current_section = 'responsible'
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')) and current_section == 'corrective':
                corrective_actions.append(line)
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')) and current_section == 'preventive':
                preventive_actions.append(line)
            elif line.startswith('- ') and current_section == 'regulatory':
                regulatory_requirements.append(line[2:])
            elif line.startswith('- ') and current_section == 'responsible':
                responsible_parties.append(line[2:])
        
        # Create analysis record
        analysis = CAPAAnalysis(
            user_id=request.user_id,
            problem_statement=request.problem_statement,
            root_causes=request.root_causes,
            corrective_actions=corrective_actions,
            preventive_actions=preventive_actions,
            regulatory_requirements=regulatory_requirements,
            timeline=timeline,
            responsible_parties=responsible_parties
        )
        
        # Store in database
        await db.capa_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        logging.error(f"CAPA analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"CAPA analysis failed: {str(e)}")

# Green Chemistry Endpoints (keeping existing)
@api_router.post("/green-chemistry/analyze", response_model=GreenChemistryAnalysis)
async def analyze_green_chemistry(request: GreenChemistryRequest):
    try:
        # Use AI to generate Green Chemistry analysis
        system_message = get_pharma_system_message(
            "GMP",  # Always use GMP for green chemistry as it's universally applicable
            specialty="green_chemistry"
        )
        
        chat = LlmChat(
            api_key=os.environ['EMERGENT_LLM_KEY'],
            session_id=f"green_chem_{uuid.uuid4()}",
            system_message=system_message
        ).with_model("openai", "gpt-4o-mini")
        
        solvents_text = ", ".join(request.current_solvents) if request.current_solvents else "Not specified"
        target_metrics_text = ", ".join(request.target_metrics) if request.target_metrics else "General sustainability improvement"
        
        prompt = f"""Perform a comprehensive Green Chemistry analysis for this pharmaceutical process:

PROCESS DESCRIPTION: {request.process_description}
CURRENT SOLVENTS: {solvents_text}
TARGET METRICS: {target_metrics_text}

Please provide:
1. SUSTAINABILITY ASSESSMENT (score 1-10, where 10 is most sustainable)
2. E-FACTOR ESTIMATION (waste generated per unit of product)
3. GREEN CHEMISTRY RECOMMENDATIONS
4. ALTERNATIVE SOLVENTS (with pros/cons)
5. PROCESS IMPROVEMENTS for sustainability
6. GREEN METRICS analysis
7. SC-CO₂ (Supercritical CO₂) opportunities if applicable

Consider the 12 Principles of Green Chemistry and pharmaceutical industry requirements.

Format your response as:
SUSTAINABILITY SCORE: [1-10 score] - [justification]

E-FACTOR: [estimated value] - [explanation]

RECOMMENDATIONS:
1. [recommendation 1]
2. [recommendation 2]
...

ALTERNATIVE SOLVENTS:
- [solvent 1]: [pros/cons]
- [solvent 2]: [pros/cons]
...

GREEN METRICS:
- Atom Economy: [assessment]
- Renewable Feedstocks: [assessment]
- Energy Efficiency: [assessment]
- Waste Reduction: [assessment]

SC-CO₂ OPPORTUNITIES:
[assessment of supercritical CO₂ applications]"""
        
        user_msg = UserMessage(text=prompt)
        ai_response = await chat.send_message(user_msg)
        
        # Parse AI response to extract structured data
        sustainability_score = None
        e_factor = None
        recommendations = []
        alternative_solvents = []
        green_metrics = {}
        
        lines = ai_response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('SUSTAINABILITY SCORE:'):
                score_text = line.replace('SUSTAINABILITY SCORE:', '').strip()
                try:
                    sustainability_score = int(score_text.split()[0])
                except:
                    sustainability_score = 5  # Default
            elif line.startswith('E-FACTOR:'):
                e_factor_text = line.replace('E-FACTOR:', '').strip()
                try:
                    e_factor = float(e_factor_text.split()[0])
                except:
                    e_factor = 10.0  # Default estimate
            elif line.startswith('RECOMMENDATIONS:'):
                current_section = 'recommendations'
            elif line.startswith('ALTERNATIVE SOLVENTS:'):
                current_section = 'solvents'
            elif line.startswith('GREEN METRICS:'):
                current_section = 'metrics'
            elif line.startswith(('1.', '2.', '3.', '4.', '5.')) and current_section == 'recommendations':
                recommendations.append(line)
            elif line.startswith('- ') and current_section == 'solvents':
                alternative_solvents.append(line[2:])
            elif line.startswith('- ') and current_section == 'metrics':
                if ':' in line:
                    key, value = line[2:].split(':', 1)
                    green_metrics[key.strip()] = value.strip()
        
        # Create analysis record
        analysis = GreenChemistryAnalysis(
            user_id=request.user_id,
            process_description=request.process_description,
            current_solvents=request.current_solvents,
            e_factor=e_factor,
            sustainability_score=sustainability_score,
            recommendations=recommendations,
            alternative_solvents=alternative_solvents,
            green_metrics=green_metrics
        )
        
        # Store in database
        await db.green_chemistry_analyses.insert_one(analysis.dict())
        
        return analysis
        
    except Exception as e:
        logging.error(f"Green chemistry analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Green chemistry analysis failed: {str(e)}")

# Get user's investigation history
@api_router.get("/investigation/history/{user_id}")
async def get_investigation_history(user_id: str):
    try:
        five_why = await db.five_why_analyses.find({"user_id": user_id}).sort("created_at", -1).to_list(10)
        fishbone = await db.fishbone_analyses.find({"user_id": user_id}).sort("created_at", -1).to_list(10)
        capa = await db.capa_analyses.find({"user_id": user_id}).sort("created_at", -1).to_list(10)
        
        return {
            "five_why": [FiveWhyAnalysis(**item) for item in five_why],
            "fishbone": [FishboneAnalysis(**item) for item in fishbone],
            "capa": [CAPAAnalysis(**item) for item in capa]
        }
    except Exception as e:
        logging.error(f"Investigation history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get investigation history: {str(e)}")

# Get user's green chemistry history
@api_router.get("/green-chemistry/history/{user_id}")
async def get_green_chemistry_history(user_id: str):
    try:
        analyses = await db.green_chemistry_analyses.find({"user_id": user_id}).sort("created_at", -1).to_list(20)
        return [GreenChemistryAnalysis(**analysis) for analysis in analyses]
    except Exception as e:
        logging.error(f"Green chemistry history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get green chemistry history: {str(e)}")

# Existing endpoints (keeping all)
@api_router.get("/chat-history/{session_id}")
async def get_chat_history(session_id: str):
    messages = await db.chat_messages.find(
        {"session_id": session_id}
    ).sort("timestamp", 1).to_list(1000)
    
    return [ChatMessage(**msg) for msg in messages]

@api_router.post("/user-profile")
async def create_user_profile(profile: UserProfile):
    profile_dict = profile.dict()
    await db.user_profiles.insert_one(profile_dict)
    return profile

@api_router.get("/user-profile/{user_id}")
async def get_user_profile(user_id: str):
    profile = await db.user_profiles.find_one({"user_id": user_id})
    if profile:
        return UserProfile(**profile)
    return None

# Process troubleshooting templates
@api_router.get("/troubleshooting-templates")
async def get_troubleshooting_templates():
    return {
        "batch_failure": {
            "name": "Batch Failure Investigation",
            "questions": [
                "What is the batch number and product?",
                "At which stage did the failure occur?",
                "What were the observed deviations?",
                "Were there any equipment malfunctions?",
                "What were the environmental conditions?"
            ],
            "regulatory_focus": ["GMP", "ICH Q10"]
        },
        "process_optimization": {
            "name": "Process Optimization",
            "questions": [
                "Which process step needs optimization?",
                "What are the current performance metrics?",
                "What are the target specifications?",
                "Are there any regulatory constraints?",
                "What resources are available?"
            ],
            "regulatory_focus": ["ICH Q8", "ICH Q9", "ICH Q10"]
        },
        "contamination_investigation": {
            "name": "Contamination Investigation",
            "questions": [
                "What type of contamination was detected?",
                "When was it first identified?",
                "Which products/batches are affected?",
                "What is the potential source?",
                "Have similar issues occurred before?"
            ],
            "regulatory_focus": ["GMP", "USFDA", "ICH Q7"]
        },
        "regulatory_proofreading": {
            "name": "Regulatory Document Proofreading",
            "questions": [
                "Is there any misleading information in this document?",
                "Review this section for regulatory compliance",
                "Help identify potential inconsistencies or errors",
                "Check for inconsistent language throughout",
                "Summarize key regulatory requirements"
            ],
            "regulatory_focus": ["GMP", "USFDA", "ICH", "EDQM", "PMDA"]
        }
    }

# Regulatory proofreading prompts endpoint
@api_router.get("/proofreading-prompts")
async def get_proofreading_prompts():
    return {
        "accuracy": [
            "Is there any misleading information in this document?",
            "Help us identify any potential inconsistencies or errors in this document."
        ],
        "compliance": [
            "Review this section of our regulatory submission and provide suggestions for improving clarity and compliance",
            "Review our clinical trial data and ensure that it aligns with regulatory guidelines and standards."
        ],
        "language": [
            "Is there any inconsistent language in this document?",
            "We need to optimize the labeling information for this drug. Can you assist with drafting clear and compliant language?"
        ],
        "summary": [
            "Summarize the key regulatory requirements for this product and provide a concise overview."
        ],
        "references": [
            "Generate a comprehensive list of references to support the efficacy and safety claims in our submission."
        ],
        "updates": [
            "Interpret the latest regulatory updates and incorporate them into our submission."
        ],
        "risk": [
            "Suggest ways to strengthen the risk management section of our regulatory documentation."
        ]
    }

# Legacy endpoints (keeping for compatibility)
@api_router.get("/")
async def root():
    return {"message": "PharmaGPT API - Ready to assist with pharmaceutical quality and R&D"}

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# Initialize sample consultant data on startup
@app.on_event("startup")
async def initialize_sample_data():
    """Initialize sample consultant data"""
    try:
        # Check if consultants already exist
        existing_consultants = await db.consultants.count_documents({})
        if existing_consultants == 0:
            sample_consultants = [
                {
                    "id": str(uuid.uuid4()),
                    "name": "Dr. Sarah Johnson",
                    "title": "Senior Regulatory Affairs Consultant",
                    "expertise_areas": ["USFDA Submissions", "GMP Compliance", "Quality Systems", "API Manufacturing"],
                    "experience_years": 15,
                    "education": ["PhD in Pharmaceutical Sciences", "MS in Chemistry"],
                    "certifications": ["RAC (Regulatory Affairs Certification)", "PMP", "ICH GCP"],
                    "bio": "Dr. Johnson has over 15 years of experience in pharmaceutical regulatory affairs, specializing in USFDA submissions and GMP compliance. She has successfully guided over 50 pharmaceutical companies through regulatory approvals.",
                    "hourly_rate": 350.0,
                    "availability_schedule": {
                        "monday": ["09:00", "10:00", "14:00", "15:00", "16:00"],
                        "tuesday": ["09:00", "10:00", "11:00", "14:00", "15:00"],
                        "wednesday": ["10:00", "11:00", "14:00", "15:00", "16:00"],
                        "thursday": ["09:00", "10:00", "14:00", "15:00"],
                        "friday": ["09:00", "10:00", "11:00", "14:00"]
                    },
                    "rating": 4.9,
                    "total_consultations": 127,
                    "languages": ["English", "Spanish"],
                    "profile_image": None,
                    "is_active": True,
                    "created_at": datetime.utcnow()
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Dr. Michael Chen",
                    "title": "API Manufacturing Expert",
                    "expertise_areas": ["API Process Development", "Process Optimization", "Green Chemistry", "Scale-up"],
                    "experience_years": 12,
                    "education": ["PhD in Chemical Engineering", "MS in Process Engineering"],
                    "certifications": ["Six Sigma Black Belt", "Project Management Professional"],
                    "bio": "Dr. Chen specializes in API manufacturing processes and has led successful scale-up projects for multiple pharmaceutical companies. Expert in green chemistry and sustainable manufacturing practices.",
                    "hourly_rate": 320.0,
                    "availability_schedule": {
                        "monday": ["08:00", "09:00", "13:00", "14:00", "15:00"],
                        "tuesday": ["08:00", "09:00", "10:00", "13:00", "14:00"],
                        "wednesday": ["09:00", "10:00", "13:00", "14:00", "15:00"],
                        "thursday": ["08:00", "09:00", "13:00", "14:00"],
                        "friday": ["08:00", "09:00", "10:00", "13:00"]
                    },
                    "rating": 4.8,
                    "total_consultations": 89,
                    "languages": ["English", "Mandarin"],
                    "profile_image": None,
                    "is_active": True,
                    "created_at": datetime.utcnow()
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Dr. Emily Rodriguez",
                    "title": "Quality Assurance Specialist",
                    "expertise_areas": ["Quality Systems", "CAPA Implementation", "Root Cause Analysis", "Validation"],
                    "experience_years": 18,
                    "education": ["PhD in Pharmaceutical Sciences", "MS in Quality Assurance"],
                    "certifications": ["ASQ CQE", "ASQ CQA", "Lead Auditor ISO 9001"],
                    "bio": "Dr. Rodriguez is a seasoned quality assurance professional with extensive experience in pharmaceutical manufacturing. She specializes in quality systems implementation and CAPA processes.",
                    "hourly_rate": 360.0,
                    "availability_schedule": {
                        "monday": ["10:00", "11:00", "15:00", "16:00"],
                        "tuesday": ["09:00", "10:00", "11:00", "15:00", "16:00"],
                        "wednesday": ["10:00", "11:00", "15:00", "16:00"],
                        "thursday": ["09:00", "10:00", "15:00", "16:00"],
                        "friday": ["10:00", "11:00", "15:00"]
                    },
                    "rating": 4.9,
                    "total_consultations": 156,
                    "languages": ["English", "Spanish", "Portuguese"],
                    "profile_image": None,
                    "is_active": True,
                    "created_at": datetime.utcnow()
                }
            ]
            
            await db.consultants.insert_many(sample_consultants)
            logger.info("Sample consultant data initialized")
    except Exception as e:
        logger.error(f"Failed to initialize sample data: {str(e)}")
