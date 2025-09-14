from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
import os
import json
import tempfile
from docx import Document
from datetime import datetime
import PyPDF2
import re
import time
import pandas as pd
import numpy as np
import sqlite3
from contextlib import contextmanager
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
CORS(app)

# Initialize rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize OpenAI (you'll need to set your API key in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Set up logging
if not app.debug:
    file_handler = RotatingFileHandler('legal_assistant.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Legal Assistant startup')

# Database setup
def get_db():
    conn = sqlite3.connect('legal_assistant.db')
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_connection():
    conn = get_db()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with db_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                history TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                text TEXT,
                filename TEXT,
                file_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT,
                ip_address TEXT,
                usage_count INTEGER DEFAULT 1,
                last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Initialize database
init_db()

# System prompt for legal assistant
LEGAL_ASSISTANT_PROMPT = """
You are JustiBot, an AI legal assistant specialized in Indian law. You provide accurate, helpful legal information while always reminding users that you are an AI and not a substitute for professional legal advice.

Key capabilities:
1. Legal drafting (contracts, petitions, notices)
2. Case law research and analysis
3. Legal document review
4. Explanation of legal concepts
5. Scenario analysis and issue spotting
6. Compliance guidance

Always:
- Be precise and cite relevant Indian laws when possible
- Maintain professional tone
- Suggest consulting actual lawyers for serious matters
- Acknowledge limitations when unsure
"""

# Enhanced legal knowledge base for fallback responses
LEGAL_KNOWLEDGE_BASE = {
    "contracts": {
        "elements": "Essential elements of a valid contract under Indian Contract Act, 1872: 1. Offer and Acceptance, 2. Lawful Consideration, 3. Capacity of Parties, 4. Free Consent, 5. Lawful Object, 6. Certainty and Possibility of Performance.",
        "types": "Common contract types in India: Service Agreement, Sale Deed, Lease Agreement, Employment Contract, Partnership Deed, Non-Disclosure Agreement (NDA), Memorandum of Understanding (MoU).",
        "breach": "Remedies for breach of contract: 1. Damages (Compensatory, Nominal, Liquidated), 2. Specific Performance, 3. Injunction, 4. Quantum Meruit, 5. Rescission."
    },
    "constitution": {
        "fundamental_rights": "Fundamental Rights under Indian Constitution (Part III): 1. Right to Equality (Articles 14-18), 2. Right to Freedom (Articles 19-22), 3. Right against Exploitation (Articles 23-24), 4. Right to Freedom of Religion (Articles 25-28), 5. Cultural and Educational Rights (Articles 29-30), 6. Right to Constitutional Remedies (Article 32).",
        "directive_principles": "Directive Principles of State Policy (Part IV) are guidelines for governance, though not enforceable in courts. They aim to establish social and economic democracy."
    },
    "ipc": {
        "common_sections": "Important IPC Sections: 302 (Murder), 304 (Culpable Homicide), 375 (Rape), 378 (Theft), 420 (Cheating), 499 (Defamation), 497 (Adultery - struck down in 2018).",
        "bailable": "Bailable offenses generally include less serious crimes where bail can be claimed as a right. Non-bailable offenses are more serious where bail is at court's discretion."
    },
    "civil_procedure": {
        "stages": "Stages of civil suit: 1. Institution of suit, 2. Issue of summons, 3. Appearance of parties, 4. Written statement, 5. Discovery and inspection, 6. Framing of issues, 7. Evidence, 8. Arguments, 9. Judgment, 10. Appeal.",
        "limitation": "The Limitation Act, 1963 prescribes time limits for different legal actions. For example: 3 years for contract disputes, 1 year for tort of defamation."
    },
    "consumer_protection": {
        "rights": "Consumer Rights under Consumer Protection Act, 2019: 1. Right to Safety, 2. Right to Information, 3. Right to Choose, 4. Right to be Heard, 5. Right to Redressal, 6. Right to Consumer Education.",
        "forums": "Consumer forums: District Commission (claims up to ₹1 crore), State Commission (claims ₹1 crore to ₹10 crores), National Commission (claims above ₹10 crores)."
    }
}

# Legal database and training data
class LegalDatabase:
    def __init__(self):
        self.cases = []
        self.statutes = []
        self.training_data = []
        self.enhanced_cases = self.load_enhanced_cases()
    
    def load_enhanced_cases(self):
        """Load enhanced case information for better fallback responses"""
        return [
            {
                "title": "Kesavananda Bharati vs State of Kerala (1973)",
                "citation": "AIR 1973 SC 1461",
                "significance": "Established the Basic Structure Doctrine - Parliament cannot amend the basic structure of the Constitution",
                "key_points": ["Basic structure doctrine", "Judicial review of amendments", "Limitation on parliamentary power"]
            },
            {
                "title": "Minerva Mills Ltd. vs Union of India (1980)",
                "citation": "AIR 1980 SC 1789",
                "significance": "Strengthened basic structure doctrine, struck down parts of 42nd Amendment",
                "key_points": ["Balance between Fundamental Rights and DPSP", "Judicial review protection"]
            },
            {
                "title": "Maneka Gandhi vs Union of India (1978)",
                "citation": "AIR 1978 SC 597",
                "significance": "Expanded scope of Article 21 (Right to Life and Personal Liberty)",
                "key_points": ["Due process", "Procedure must be fair, just and reasonable", "Expanded personal liberty"]
            }
        ]
    
    def load_cases_from_csv(self, filepath):
        try:
            # Create data directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Check if file exists, if not create a sample one
            if not os.path.exists(filepath):
                sample_cases = pd.DataFrame([
                    {
                        "case_no": "AIR 1973 SC 1461",
                        "title": "Kesavananda Bharati vs State of Kerala",
                        "year": 1973,
                        "court": "Supreme Court of India",
                        "summary": "Landmark case that established the Basic Structure Doctrine of the Constitution",
                        "key_issues": "Constitutional amendments, Fundamental rights, Basic structure doctrine",
                        "outcome": "The Supreme Court outlined the basic structure doctrine of the Constitution"
                    },
                    {
                        "case_no": "AIR 1980 SC 1789",
                        "title": "Minerva Mills Ltd. vs Union of India",
                        "year": 1980,
                        "court": "Supreme Court of India",
                        "summary": "Strengthened the basic structure doctrine laid down in Kesavananda Bharati case",
                        "key_issues": "Constitutional amendments, Judicial review, Fundamental rights",
                        "outcome": "Struck down parts of the 42nd Amendment that prevented judicial review of constitutional amendments"
                    }
                ])
                sample_cases.to_csv(filepath, index=False)
                self.cases = sample_cases.to_dict('records')
            else:
                df = pd.read_csv(filepath)
                self.cases = df.to_dict('records')
            return len(self.cases)
        except Exception as e:
            app.logger.error(f"Error loading cases: {e}")
            # Return dummy data if file not found
            self.cases = [
                {
                    "case_no": "AIR 1973 SC 1461",
                    "title": "Kesavananda Bharati vs State of Kerala",
                    "year": 1973,
                    "court": "Supreme Court of India",
                    "summary": "Landmark case that established the Basic Structure Doctrine of the Constitution",
                    "key_issues": "Constitutional amendments, Fundamental rights, Basic structure doctrine",
                    "outcome": "The Supreme Court outlined the basic structure doctrine of the Constitution"
                },
                {
                    "case_no": "AIR 1980 SC 1789",
                    "title": "Minerva Mills Ltd. vs Union of India",
                    "year": 1980,
                    "court": "Supreme Court of India",
                    "summary": "Strengthened the basic structure doctrine laid down in Kesavananda Bharati case",
                    "key_issues": "Constitutional amendments, Judicial review, Fundamental rights",
                    "outcome": "Struck down parts of the 42nd Amendment that prevented judicial review of constitutional amendments"
                }
            ]
            return len(self.cases)
    
    def load_training_data(self):
        # Pre-load training examples for the AI
        self.training_data = [
            {
                "input": "How to draft a legal notice?",
                "output": "A legal notice should include: 1. sender and recipient details, 2. clear subject line, 3. facts of the case, 4. legal basis for the claim, 5. relief sought, 6. time given for compliance, and 7. consequences of non-compliance. Would you like me to help draft a specific notice?"
            },
            {
                "input": "What is the difference between civil and criminal law?",
                "output": "Civil law deals with disputes between individuals/organizations where compensation may be awarded. Criminal law deals with crimes against the state/society where punishment may be imposed. Key differences: purpose (compensation vs punishment), burden of proof (balance of probabilities vs beyond reasonable doubt), and initiating party (individual vs state)."
            },
            {
                "input": "How to file a consumer complaint?",
                "output": "To file a consumer complaint: 1. Gather all documents ( bills, correspondence, evidence), 2. Draft a complaint with facts and relief sought, 3. File with the appropriate consumer forum based on the value of claim, 4. Pay the required fee. The process is designed to be consumer-friendly and doesn't necessarily require a lawyer."
            }
        ]
        return len(self.training_data)

# Initialize legal database
legal_db = LegalDatabase()

def preload_training_data():
    """Pre-load training data and cases at startup"""
    app.logger.info("Loading legal cases...")
    num_cases = legal_db.load_cases_from_csv("data/legal_cases.csv")
    
    app.logger.info("Loading training data...")
    num_training = legal_db.load_training_data()
    
    app.logger.info(f"Loaded {num_cases} legal cases and {num_training} training examples")
    return num_cases, num_training

def track_api_usage(endpoint):
    """Track API usage for rate limiting and analytics"""
    ip_address = get_remote_address()
    with db_connection() as conn:
        # Check if this IP has used this endpoint recently
        cursor = conn.execute(
            "SELECT * FROM api_usage WHERE endpoint = ? AND ip_address = ?",
            (endpoint, ip_address)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            conn.execute(
                "UPDATE api_usage SET usage_count = usage_count + 1, last_used = CURRENT_TIMESTAMP WHERE id = ?",
                (existing['id'],)
            )
        else:
            # Create new record
            conn.execute(
                "INSERT INTO api_usage (endpoint, ip_address) VALUES (?, ?)",
                (endpoint, ip_address)
            )
        conn.commit()

def extract_text_from_pdf(file_content):
    """Improved PDF text extraction function"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        text = ""
        with open(tmp_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        
        os.unlink(tmp_path)
        return text
    except Exception as e:
        app.logger.error(f"PDF extraction error: {e}")
        return ""

def get_fallback_response(user_message):
    """Generate a fallback response when OpenAI API is unavailable"""
    user_message_lower = user_message.lower()
    
    # Greeting responses
    if any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm JustiBot, your AI legal assistant. I'm currently operating in limited mode. How can I help you with Indian legal information today?"
    
    # Contract-related queries
    elif any(word in user_message_lower for word in ['contract', 'draft', 'agreement']):
        return f"{LEGAL_KNOWLEDGE_BASE['contracts']['elements']}\n\n{LEGAL_KNOWLEDGE_BASE['contracts']['types']}"
    
    # Case law queries
    elif any(word in user_message_lower for word in ['case', 'judgment', 'precedent']):
        case_info = "\n\n".join([f"{case['title']} ({case['citation']}): {case['significance']}" 
                               for case in legal_db.enhanced_cases[:2]])
        return f"Here are some important Indian legal cases:\n\n{case_info}"
    
    # Fundamental rights queries
    elif any(word in user_message_lower for word in ['right', 'fundamental', 'constitution']):
        return LEGAL_KNOWLEDGE_BASE['constitution']['fundamental_rights']
    
    # Consumer protection queries
    elif any(word in user_message_lower for word in ['consumer', 'complaint', 'forum']):
        return f"{LEGAL_KNOWLEDGE_BASE['consumer_protection']['rights']}\n\n{LEGAL_KNOWLEDGE_BASE['consumer_protection']['forums']}"
    
    # IPC queries
    elif any(word in user_message_lower for word in ['ipc', 'section', 'penal']):
        return LEGAL_KNOWLEDGE_BASE['ipc']['common_sections']
    
    # General legal help
    else:
        return ("I'm currently operating in limited mode due to API restrictions. "
                "I can help with information about: Indian Contract Law, Constitution, "
                "IPC sections, Consumer Protection, and important legal cases. "
                "Please ask me specific legal questions about Indian law.")

@app.route('/')
def serve_index():
    try:
        return send_file('index.html')
    except:
        return "HTML file not found. Please make sure index.html is in the same directory."

@app.route('/openai/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat_with_openai():
    try:
        track_api_usage('/openai/chat')
        
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])
        session_id = data.get('session_id', 'default')
        model = data.get('model', 'gpt-3.5-turbo')
        
        # Save chat history to database
        with db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO chat_sessions (id, history, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (session_id, json.dumps(history + [{"role": "user", "content": user_message}]))
            )
            conn.commit()
        
        # Prepare messages with system prompt and history
        messages = [{"role": "system", "content": LEGAL_ASSISTANT_PROMPT}]
        
        # Add history if provided
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        ai_response = ""
        
        # Try to call OpenAI API, use fallback if it fails
        if client:
            try:
                # Call OpenAI API with the new client format
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=900
                )
                
                ai_response = response.choices[0].message.content
                
            except Exception as api_error:
                app.logger.error(f"OpenAI API error, using fallback: {api_error}")
                ai_response = get_fallback_response(user_message)
                ai_response += "\n\n[Note: Currently using fallback mode due to API limitations]"
        else:
            ai_response = get_fallback_response(user_message)
            ai_response += "\n\n[Note: Operating in fallback mode - OpenAI API not configured]"
        
        # Update chat history with AI response
        with db_connection() as conn:
            conn.execute(
                "UPDATE chat_sessions SET history = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (json.dumps(history + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ai_response}
                ]), session_id)
            )
            conn.commit()
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        # Return fallback response even for other errors
        fallback_response = get_fallback_response(user_message if 'user_message' in locals() else "")
        return jsonify({"response": fallback_response})

# [Keep all your other routes the same as before - /upload, /analyze-documents, /draft-document, etc.]
# They will need similar fallback mechanisms, but for brevity, I'll show the pattern for one more:

@app.route('/draft-document', methods=['POST'])
@limiter.limit("10 per hour")
def draft_document():
    try:
        track_api_usage('/draft-document')
        
        data = request.json
        doc_type = data.get('type', 'contract')
        requirements = data.get('requirements', '')
        jurisdiction = data.get('jurisdiction', 'India')
        
        # If OpenAI is not available, provide a template
        if not client:
            return jsonify({
                "draft": f"""# {doc_type.title()} Draft - Template
                
[This is a template as OpenAI service is currently unavailable]

**PARTIES:**
[Insert Party Names and Details]

**RECITALS:**
[Background and purpose of the {doc_type}]

**TERMS AND CONDITIONS:**

1. [Insert first term]
2. [Insert second term]
3. [Insert third term]

**GOVERNING LAW:**
This {doc_type} shall be governed by and construed in accordance with the laws of {jurisdiction}.

**SIGNATURES:**

_________________________
[Party A Name and Title]

_________________________
[Party B Name and Title]

**Note:** This is a template. Please consult with a qualified legal professional for specific advice."""
            })
        
        # Prepare drafting prompt
        draft_prompt = f"""
        Draft a {doc_type} for Indian jurisdiction with the following requirements:
        
        {requirements}
        
        Please provide a comprehensive, professionally formatted legal document with:
        1. Appropriate headings and sections
        2. Standard legal language
        3. Placeholders for specific details in [brackets]
        4. Relevant legal provisions for {jurisdiction}
        """
        
        # Call OpenAI for drafting
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": draft_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        draft = response.choices[0].message.content
        
        return jsonify({"draft": draft})
    
    except Exception as e:
        app.logger.error(f"Drafting error: {e}")
        # Provide a basic template as fallback
        return jsonify({
            "draft": f"# Document Draft Template\n\nI encountered an error generating your document. Here's a basic template structure for a {data.get('type', 'contract')}:\n\n1. Parties Section\n2. Definitions\n3. Terms and Conditions\n4. Payment Terms (if applicable)\n5. Termination Clause\n6. Governing Law\n7. Signatures\n\nPlease try again later or consult a legal professional for specific drafting needs."
        })

# [All your other routes should follow this pattern with fallback responses]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "uptime": time.time() - app.config.get("START_TIME", time.time()),
        "cases_loaded": len(legal_db.cases),
        "training_examples": len(legal_db.training_data),
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_available": client is not None,
        "fallback_mode": client is None or not OPENAI_API_KEY
    })

if __name__ == "__main__":
    # Pre-load some training data
    preload_training_data()
    
    app.config["START_TIME"] = time.time()
    app.logger.info("JustiBot AI Legal Assistant starting...")
    app.logger.info(f"OpenAI API Key: {'Loaded' if OPENAI_API_KEY else 'Missing'}")
    app.logger.info(f"OpenAI Client: {'Available' if client else 'Not available'}")
    app.logger.info(f"Loaded {len(legal_db.cases)} legal cases")
    app.logger.info(f"Pre-loaded {len(legal_db.training_data)} training examples")
    
    if not client or not OPENAI_API_KEY:
        app.logger.warning("Running in fallback mode - OpenAI API not available")
    
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)