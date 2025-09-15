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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

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
        self.vectorizer = None
        self.justice_dataset = None
        self.tfidf_matrix = None
        self.case_vectorizer = None
        self.case_tfidf_matrix = None
    
    def load_justice_dataset(self):
        """Load and process the justice.csv dataset with proper column handling"""
        try:
            justice_path = "legal_datasets/justice.csv"
            if os.path.exists(justice_path):
                self.justice_dataset = pd.read_csv(justice_path)
                app.logger.info(f"Loaded justice dataset with {len(self.justice_dataset)} entries")
                
                # Create searchable text from available columns
                text_columns = []
                if 'name' in self.justice_dataset.columns:
                    text_columns.append('name')
                if 'facts' in self.justice_dataset.columns:
                    text_columns.append('facts')
                if 'docket' in self.justice_dataset.columns:
                    text_columns.append('docket')
                if 'first_party' in self.justice_dataset.columns:
                    text_columns.append('first_party')
                if 'second_party' in self.justice_dataset.columns:
                    text_columns.append('second_party')
                if 'issue_area' in self.justice_dataset.columns:
                    text_columns.append('issue_area')
                
                if text_columns:
                    # Combine relevant columns for search
                    self.justice_dataset['search_text'] = self.justice_dataset[text_columns].fillna('').apply(
                        lambda row: ' '.join(row.values.astype(str)), axis=1
                    )
                    
                    # Create TF-IDF vectorizer
                    self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
                    self.tfidf_matrix = self.vectorizer.fit_transform(self.justice_dataset['search_text'])
                    
                    # Save the vectorizer for later use
                    os.makedirs("models", exist_ok=True)
                    joblib.dump(self.vectorizer, 'models/tfidf_vectorizer.joblib')
                    joblib.dump(self.tfidf_matrix, 'models/tfidf_matrix.joblib')
                    joblib.dump(self.justice_dataset, 'models/justice_dataset.joblib')
                    
                    app.logger.info("Justice dataset processed and vectorized successfully")
                    
                    # Also create case-specific search for legal cases
                    self.prepare_case_search()
                else:
                    app.logger.warning("No suitable columns found in justice dataset for search")
            else:
                app.logger.warning("Justice dataset not found at legal_datasets/justice.csv")
                # Create sample dataset if not exists
                os.makedirs("legal_datasets", exist_ok=True)
                sample_data = pd.DataFrame({
                    'name': [
                        'Kesavananda Bharati vs State of Kerala',
                        'Maneka Gandhi vs Union of India',
                        'Minerva Mills vs Union of India'
                    ],
                    'docket': ['AIR 1973 SC 1461', 'AIR 1978 SC 597', 'AIR 1980 SC 1789'],
                    'facts': [
                        'Landmark case that established the Basic Structure Doctrine of the Constitution',
                        'Expanded the scope of Article 21 (Right to Life and Personal Liberty)',
                        'Strengthened the basic structure doctrine laid down in Kesavananda Bharati case'
                    ],
                    'issue_area': ['Constitutional Law', 'Fundamental Rights', 'Constitutional Law']
                })
                sample_data.to_csv(justice_path, index=False)
                self.justice_dataset = sample_data
                app.logger.info("Created sample justice dataset")
                
        except Exception as e:
            app.logger.error(f"Error loading justice dataset: {e}")
    
    def prepare_case_search(self):
        """Prepare case-specific search using the dataset columns"""
        try:
            if self.justice_dataset is not None:
                # Create case text from available columns
                case_texts = []
                for _, row in self.justice_dataset.iterrows():
                    case_text = ""
                    if 'name' in row and pd.notna(row['name']):
                        case_text += f"Case: {row['name']}. "
                    if 'docket' in row and pd.notna(row['docket']):
                        case_text += f"Docket: {row['docket']}. "
                    if 'facts' in row and pd.notna(row['facts']):
                        case_text += f"Facts: {row['facts']}. "
                    if 'issue_area' in row and pd.notna(row['issue_area']):
                        case_text += f"Issue: {row['issue_area']}. "
                    if 'first_party' in row and pd.notna(row['first_party']):
                        case_text += f"Parties: {row['first_party']}"
                        if 'second_party' in row and pd.notna(row['second_party']):
                            case_text += f" vs {row['second_party']}. "
                    
                    case_texts.append(case_text)
                
                self.justice_dataset['case_text'] = case_texts
                
                # Create case-specific vectorizer
                self.case_vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
                self.case_tfidf_matrix = self.case_vectorizer.fit_transform(self.justice_dataset['case_text'])
                
                joblib.dump(self.case_vectorizer, 'models/case_vectorizer.joblib')
                joblib.dump(self.case_tfidf_matrix, 'models/case_tfidf_matrix.joblib')
                
                app.logger.info("Case search prepared successfully")
                
        except Exception as e:
            app.logger.error(f"Error preparing case search: {e}")
    
    def find_similar_cases(self, query, top_n=5):
        """Find similar cases based on the query"""
        if self.case_vectorizer is None or self.case_tfidf_matrix is None:
            return []
        
        try:
            # Vectorize the query
            query_vec = self.case_vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vec, self.case_tfidf_matrix).flatten()
            
            # Get top N most similar cases
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    row = self.justice_dataset.iloc[idx]
                    case_info = {
                        'similarity': float(similarities[idx]),
                        'rank': len(results) + 1
                    }
                    
                    # Add available information
                    if 'name' in row and pd.notna(row['name']):
                        case_info['name'] = row['name']
                    if 'docket' in row and pd.notna(row['docket']):
                        case_info['docket'] = row['docket']
                    if 'href' in row and pd.notna(row['href']):
                        case_info['href'] = row['href']
                    if 'facts' in row and pd.notna(row['facts']):
                        case_info['facts'] = row['facts'][:200] + "..." if len(str(row['facts'])) > 200 else row['facts']
                    if 'issue_area' in row and pd.notna(row['issue_area']):
                        case_info['issue_area'] = row['issue_area']
                    if 'first_party' in row and pd.notna(row['first_party']):
                        case_info['parties'] = row['first_party']
                        if 'second_party' in row and pd.notna(row['second_party']):
                            case_info['parties'] += f" vs {row['second_party']}"
                    if 'majority_vote' in row and pd.notna(row['majority_vote']):
                        case_info['majority_vote'] = row['majority_vote']
                    if 'decision_type' in row and pd.notna(row['decision_type']):
                        case_info['decision_type'] = row['decision_type']
                    
                    results.append(case_info)
            
            return results
        except Exception as e:
            app.logger.error(f"Error finding similar cases: {e}")
            return []
    
    def find_similar_questions(self, query, top_n=3):
        """Find similar questions in the justice dataset"""
        if self.vectorizer is None or self.tfidf_matrix is None:
            return []
        
        try:
            # Vectorize the query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Get top N most similar questions
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    row = self.justice_dataset.iloc[idx]
                    result = {
                        'similarity': float(similarities[idx])
                    }
                    
                    # Add available information
                    if 'name' in row and pd.notna(row['name']):
                        result['case_name'] = row['name']
                    if 'docket' in row and pd.notna(row['docket']):
                        result['docket'] = row['docket']
                    if 'facts' in row and pd.notna(row['facts']):
                        result['facts'] = row['facts'][:150] + "..." if len(str(row['facts'])) > 150 else row['facts']
                    if 'issue_area' in row and pd.notna(row['issue_area']):
                        result['issue_area'] = row['issue_area']
                    
                    results.append(result)
            
            return results
        except Exception as e:
            app.logger.error(f"Error finding similar questions: {e}")
            return []
    
    def get_case_details(self, case_id=None, case_name=None, docket=None):
        """Get detailed information about a specific case"""
        if self.justice_dataset is None:
            return None
        
        try:
            if case_id is not None:
                result = self.justice_dataset[self.justice_dataset['ID'] == case_id]
            elif case_name is not None:
                result = self.justice_dataset[self.justice_dataset['name'].str.contains(case_name, case=False, na=False)]
            elif docket is not None:
                result = self.justice_dataset[self.justice_dataset['docket'].str.contains(docket, case=False, na=False)]
            else:
                return None
            
            if len(result) > 0:
                case = result.iloc[0].to_dict()
                # Clean up the data
                clean_case = {}
                for key, value in case.items():
                    if pd.notna(value):
                        clean_case[key] = value
                return clean_case
            
            return None
        except Exception as e:
            app.logger.error(f"Error getting case details: {e}")
            return None
    
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
    
    app.logger.info("Loading justice dataset...")
    legal_db.load_justice_dataset()
    
    app.logger.info(f"Loaded {num_cases} legal cases, {num_training} training examples, and justice dataset")
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
    
    # First try to find similar cases in justice dataset
    similar_cases = legal_db.find_similar_cases(user_message, top_n=3)
    if similar_cases and similar_cases[0]['similarity'] > 0.3:
        best_case = similar_cases[0]
        response = "I found a relevant legal case:\n\n"
        
        if 'name' in best_case:
            response += f"**Case Name:** {best_case['name']}\n"
        if 'docket' in best_case:
            response += f"**Citation:** {best_case['docket']}\n"
        if 'facts' in best_case:
            response += f"**Summary:** {best_case['facts']}\n"
        if 'issue_area' in best_case:
            response += f"**Issue Area:** {best_case['issue_area']}\n"
        if 'parties' in best_case:
            response += f"**Parties:** {best_case['parties']}\n"
        
        response += f"\n[Similarity: {best_case['similarity']:.2f}]"
        return response
    
    # Greeting responses
    if any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'greetings']):
        return "Hello! I'm JustiBot, your AI legal assistant. I'm currently operating in limited mode. How can I help you with Indian legal information today?"
    
    # Contract-related queries
    elif any(word in user_message_lower for word in ['contract', 'draft', 'agreement']):
        return f"{LEGAL_KNOWLEDGE_BASE['contracts']['elements']}\n\n{LEGAL_KNOWLEDGE_BASE['contracts']['types']}"
    
    # Case law queries
    elif any(word in user_message_lower for word in ['case', 'judgment', 'precedent', 'supreme court', 'high court']):
        case_info = "\n\n".join([f"{case['title']} ({case['citation']}): {case['significance']}" 
                               for case in legal_db.enhanced_cases[:2]])
        return f"Here are some important Indian legal cases:\n\n{case_info}"
    
    # Fundamental rights queries
    elif any(word in user_message_lower for word in ['right', 'fundamental', 'constitution', 'article 14', 'article 21']):
        return LEGAL_KNOWLEDGE_BASE['constitution']['fundamental_rights']
    
    # Consumer protection queries
    elif any(word in user_message_lower for word in ['consumer', 'complaint', 'forum']):
        return f"{LEGAL_KNOWLEDGE_BASE['consumer_protection']['rights']}\n\n{LEGAL_KNOWLEDGE_BASE['consumer_protection']['forums']}"
    
    # IPC queries
    elif any(word in user_message_lower for word in ['ipc', 'section', 'penal', 'criminal']):
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

@app.route('/api/justice-search', methods=['POST'])
def justice_search():
    """Search for similar cases in the justice dataset"""
    try:
        data = request.json
        query = data.get('query', '')
        top_n = data.get('top_n', 5)
        
        similar_cases = legal_db.find_similar_cases(query, top_n)
        
        return jsonify({
            "query": query,
            "results": similar_cases,
            "total_results": len(similar_cases)
        })
    
    except Exception as e:
        app.logger.error(f"Justice search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/case-details', methods=['GET'])
def get_case_details():
    """Get detailed information about a specific case"""
    try:
        case_id = request.args.get('id')
        case_name = request.args.get('name')
        docket = request.args.get('docket')
        
        case_details = legal_db.get_case_details(case_id, case_name, docket)
        
        if case_details:
            return jsonify(case_details)
        else:
            return jsonify({"error": "Case not found"}), 404
    
    except Exception as e:
        app.logger.error(f"Case details error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/justice-dataset', methods=['GET'])
def get_justice_dataset():
    """Get information about the justice dataset"""
    try:
        if legal_db.justice_dataset is None:
            return jsonify({"error": "Justice dataset not loaded"}), 404
        
        dataset_info = {
            "total_entries": len(legal_db.justice_dataset),
            "columns": list(legal_db.justice_dataset.columns),
            "sample_cases": []
        }
        
        # Add sample cases if available
        if 'name' in legal_db.justice_dataset.columns:
            sample_cases = legal_db.justice_dataset.head(5)[['name', 'docket']].to_dict('records')
            dataset_info["sample_cases"] = sample_cases
        
        return jsonify(dataset_info)
    
    except Exception as e:
        app.logger.error(f"Justice dataset info error: {e}")
        return jsonify({"error": str(e)}), 500

# [Keep all your other routes the same - /upload, /analyze-documents, /draft-document, etc.]

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "uptime": time.time() - app.config.get("START_TIME", time.time()),
        "cases_loaded": len(legal_db.cases),
        "training_examples": len(legal_db.training_data),
        "justice_dataset_loaded": legal_db.justice_dataset is not None,
        "justice_dataset_size": len(legal_db.justice_dataset) if legal_db.justice_dataset is not None else 0,
        "openai_configured": bool(OPENAI_API_KEY),
        "openai_available": client is not None,
        "fallback_mode": client is None or not OPENAI_API_KEY
    })

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("legal_datasets", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("backups", exist_ok=True)
    
    # Pre-load training data
    preload_training_data()
    
    app.config["START_TIME"] = time.time()
    app.logger.info("JustiBot AI Legal Assistant starting...")
    app.logger.info(f"OpenAI API Key: {'Loaded' if OPENAI_API_KEY else 'Missing'}")
    app.logger.info(f"OpenAI Client: {'Available' if client else 'Not available'}")
    app.logger.info(f"Loaded {len(legal_db.cases)} legal cases")
    app.logger.info(f"Pre-loaded {len(legal_db.training_data)} training examples")
    app.logger.info(f"Justice dataset: {'Loaded' if legal_db.justice_dataset is not None else 'Not available'}")
    
    if not client or not OPENAI_API_KEY:
        app.logger.warning("Running in fallback mode - OpenAI API not available")
    
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)