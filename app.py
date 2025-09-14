from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

# Initialize OpenAI (you'll need to set your API key in environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# In-memory storage for demo purposes (use a database in production)
chat_sessions = {}
uploaded_documents = {}

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

# Legal database and training data
class LegalDatabase:
    def __init__(self):
        self.cases = []
        self.statutes = []
        self.training_data = []
    
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
            print(f"Error loading cases: {e}")
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
    print("Loading legal cases...")
    num_cases = legal_db.load_cases_from_csv("data/legal_cases.csv")
    
    print("Loading training data...")
    num_training = legal_db.load_training_data()
    
    print(f"Loaded {num_cases} legal cases and {num_training} training examples")
    return num_cases, num_training

@app.route('/')
def serve_index():
    try:
        return send_file('index.html')
    except:
        return "HTML file not found. Please make sure index.html is in the same directory."

@app.route('/openai/chat', methods=['POST'])
def chat_with_openai():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        user_message = data.get('message', '')
        history = data.get('history', [])
        model = data.get('model', 'gpt-3.5-turbo')
        
        # Prepare messages with system prompt and history
        messages = [{"role": "system", "content": LEGAL_ASSISTANT_PROMPT}]
        
        # Add history if provided
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        # Call OpenAI API with the new client format
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=900
        )
        
        ai_response = response.choices[0].message.content
        
        return jsonify({"response": ai_response})
    
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return jsonify({"error": "I'm having trouble connecting to the AI service. Please try again."}), 500

@app.route('/openai/chat-stream', methods=['POST'])
def chat_with_openai_stream():
    # This would implement streaming responses
    # For simplicity, we'll use the non-streaming version in this demo
    return chat_with_openai()

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        data = request.json
        text = data.get('text', '')
        session_id = data.get('session_id', 'default')
        
        if session_id not in uploaded_documents:
            uploaded_documents[session_id] = []
        
        # Store document (in production, you'd store this properly)
        uploaded_documents[session_id].append({
            'text': text[:5000],  # Limit size for demo
            'timestamp': datetime.now().isoformat()
        })
        
        return jsonify({"status": "success", "message": "Document uploaded successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-documents', methods=['POST'])
def analyze_documents():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        session_id = data.get('session_id', 'default')
        question = data.get('question', '')
        
        if session_id not in uploaded_documents or not uploaded_documents[session_id]:
            return jsonify({"error": "No documents found for analysis"}), 400
        
        # Get all uploaded documents
        documents_text = "\n\n".join([doc['text'] for doc in uploaded_documents[session_id]])
        
        # Prepare analysis prompt
        analysis_prompt = f"""
        Analyze the following legal documents and answer the user's question.
        
        DOCUMENTS:
        {documents_text}
        
        QUESTION: {question}
        
        Provide a comprehensive analysis with:
        1. Key findings from the documents
        2. Relevant legal provisions
        3. Potential issues or concerns
        4. Recommendations
        """
        
        # Call OpenAI for analysis
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        
        return jsonify({"analysis": analysis})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/draft-document', methods=['POST'])
def draft_document():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        doc_type = data.get('type', 'contract')
        requirements = data.get('requirements', '')
        jurisdiction = data.get('jurisdiction', 'India')
        
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
        return jsonify({"error": str(e)}), 500

@app.route('/find-judgments', methods=['POST'])
def find_judgments():
    try:
        data = request.json
        issue = data.get('issue', '')
        jurisdiction = data.get('jurisdiction', 'Supreme Court of India')
        limit = data.get('limit', 5)
        
        # First, try to find matches in our database
        matching_cases = []
        for case in legal_db.cases:
            if (issue.lower() in case.get('key_issues', '').lower() or 
                issue.lower() in case.get('summary', '').lower() or
                issue.lower() in case.get('title', '').lower()):
                matching_cases.append(case)
                if len(matching_cases) >= limit:
                    break
        
        # If we found cases in our database, return them
        if matching_cases:
            return jsonify({
                "judgments": matching_cases,
                "source": "local_database"
            })
        
        # If no local matches and OpenAI is available, use it to find relevant judgments
        if client:
            judgment_prompt = f"""
            Based on your knowledge of Indian case law, provide information about relevant judgments for the following legal issue:
            
            {issue}
            
            Please include:
            1. Key case names and citations
            2. Summary of legal principles established
            3. Relevance to the issue presented
            4. Any limitations or subsequent developments
            """
            
            # Call OpenAI for judgment search
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                    {"role": "user", "content": judgment_prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            judgments = response.choices[0].message.content
            
            return jsonify({
                "judgments": judgments,
                "source": "openai"
            })
        else:
            return jsonify({
                "judgments": "OpenAI service not available. Please configure your API key.",
                "source": "error"
            })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/export-docx', methods=['POST'])
def export_docx():
    try:
        data = request.json
        content = data.get('content', '')
        filename = data.get('filename', 'legal_document')
        
        # Create a new Document
        doc = Document()
        
        # Add content to the document
        for line in content.split('\n'):
            if line.strip() == '':
                continue
            # Check if line looks like a heading
            if re.match(r'^[A-Z][A-Za-z\s]+:$', line) or re.match(r'^[IVX]+\.', line) or re.match(r'^[0-9]+\.', line):
                heading = doc.add_heading(line, level=2)
            else:
                paragraph = doc.add_paragraph(line)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.docx')
        doc.save(temp_file.name)
        temp_file.close()
        
        return send_file(
            temp_file.name, 
            as_attachment=True, 
            download_name=f"{filename}.docx",
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict-outcome', methods=['POST'])
def predict_outcome():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        case_details = data.get('case_details', '')
        jurisdiction = data.get('jurisdiction', 'India')
        
        # Prepare prediction prompt
        prediction_prompt = f"""
        Based on the following case details, provide a predictive analysis of likely outcomes:
        
        {case_details}
        
        Please include:
        1. Assessment of strengths and weaknesses of the case
        2. Relevant legal precedents
        3. Potential arguments for each side
        4. Likely outcomes with probabilities
        5. Factors that could influence the outcome
        
        Remember to emphasize that this is predictive analysis only and not legal advice.
        """
        
        # Call OpenAI for prediction
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": prediction_prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        prediction = response.choices[0].message.content
        
        return jsonify({"prediction": prediction})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compliance-check', methods=['POST'])
def compliance_check():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        document_text = data.get('document_text', '')
        regulations = data.get('regulations', 'Indian laws')
        
        # Prepare compliance check prompt
        compliance_prompt = f"""
        Review the following document for compliance with {regulations}:
        
        {document_text}
        
        Please provide:
        1. Identification of potential compliance issues
        2. Relevant legal requirements
        3. Recommendations for addressing any issues
        4. References to specific regulations or standards
        """
        
        # Call OpenAI for compliance check
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": compliance_prompt}
            ],
            temperature=0.2,
            max_tokens=1500
        )
        
        compliance_report = response.choices[0].message.content
        
        return jsonify({"compliance_report": compliance_report})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/due-diligence', methods=['POST'])
def due_diligence():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        documents_text = data.get('documents_text', '')
        transaction_type = data.get('transaction_type', 'general')
        
        # Prepare due diligence prompt
        diligence_prompt = f"""
        Perform due diligence analysis on the following documents for a {transaction_type} transaction:
        
        {documents_text}
        
        Please provide:
        1. Identification of key risks and issues
        2. Legal implications of identified issues
        3. Recommendations for risk mitigation
        4. Priority areas for further investigation
        5. Potential impact on transaction structure
        """
        
        # Call OpenAI for due diligence
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": diligence_prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        diligence_report = response.choices[0].message.content
        
        return jsonify({"diligence_report": diligence_report})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/deposition-prep', methods=['POST'])
def deposition_prep():
    try:
        if not client:
            return jsonify({"error": "OpenAI API key not configured"}), 500
            
        data = request.json
        case_details = data.get('case_details', '')
        witness_role = data.get('witness_role', 'general')
        
        # Prepare deposition preparation prompt
        deposition_prompt = f"""
        Prepare deposition questions for a {witness_role} witness based on the following case details:
        
        {case_details}
        
        Please provide:
        1. Background and foundational questions
        2. Key factual questions specific to the case
        3. Questions to establish or challenge credibility
        4. Questions about documents or evidence
        5. Potential follow-up questions based on likely responses
        6. Strategy notes for the examining attorney
        """
        
        # Call OpenAI for deposition preparation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": LEGAL_ASSISTANT_PROMPT},
                {"role": "user", "content": deposition_prompt}
            ],
            temperature=0.3,
            max_tokens=1800
        )
        
        deposition_questions = response.choices[0].message.content
        
        return jsonify({"deposition_questions": deposition_questions})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "uptime": time.time() - app.config.get("START_TIME", time.time()),
        "cases_loaded": len(legal_db.cases),
        "training_examples": len(legal_db.training_data),
        "openai_configured": bool(OPENAI_API_KEY)
    })

if __name__ == "__main__":
    # Pre-load some training data
    preload_training_data()
    
    app.config["START_TIME"] = time.time()
    print("JustiBot AI Legal Assistant starting...")
    print(f"OpenAI API Key: {'Loaded' if OPENAI_API_KEY else 'Missing'}")
    print(f"Loaded {len(legal_db.cases)} legal cases")
    print(f"Pre-loaded {len(legal_db.training_data)} training examples")
    
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)