import os
import re
import uuid
import time
import numpy as np
import pandas as pd
import spacy
import requests
from html import unescape
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from functools import lru_cache
import werkzeug.exceptions as wz
from openai import OpenAI
import json
from typing import List, Dict, Any

# -----------------------------------------------------------------------------
# Environment / OpenAI
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")
client = OpenAI()  # reads key from env by default

# -----------------------------------------------------------------------------
# Flask app
# -----------------------------------------------------------------------------
app = Flask(__name__, static_folder='.')
CORS(app)  # enable CORS for browser clients
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# -----------------------------------------------------------------------------
# Error handlers
# -----------------------------------------------------------------------------
@app.errorhandler(wz.BadRequest)
def handle_400(e):
    return jsonify({"error": "bad_request", "message": str(e)}), 400

@app.errorhandler(wz.NotFound)
def handle_404(e):
    return jsonify({"error": "not_found", "message": "Route not found"}), 404

@app.errorhandler(Exception)
def handle_500(e):
    return jsonify({"error": "server_error", "message": str(e)}), 500

# -----------------------------------------------------------------------------
# Legal Database APIs (Indian Legal Resources)
# -----------------------------------------------------------------------------
class LegalDatabase:
    def __init__(self):
        self.sources = {
            'indiankanoon': 'https://api.indiankanoon.org',
            'supremecourt': 'https://api.main.sci.gov.in',
            'manupatra': 'https://api.manupatra.com'
        }
        self.training_data = []
    
    def search_live_cases(self, query, court=None, year=None):
        """Search live cases from various Indian legal databases"""
        results = []
        
        try:
            # Indian Kanoon API (public cases) - this is a placeholder
            # Actual implementation would require proper API access
            print(f"Would search live cases for: {query}")
            
            # Simulate some recent case results for demonstration
            current_year = datetime.now().year
            if "constitution" in query.lower():
                results.append({
                    'source': 'Simulated Recent Case',
                    'title': 'Recent Constitutional Matter (2024)',
                    'citation': 'AIR 2024 SC 1',
                    'date': '2024-01-15',
                    'summary': 'A recent constitutional interpretation case addressing fundamental rights',
                    'url': '#'
                })
            
        except Exception as e:
            print(f"Live search error: {e}")
        
        return results

    def add_training_data(self, question: str, answer: str, category: str = "general"):
        """Add training data to improve responses"""
        self.training_data.append({
            "question": question,
            "answer": answer,
            "category": category,
            "timestamp": datetime.now().isoformat()
        })
        return len(self.training_data)

    def get_training_data(self, category: str = None):
        """Retrieve training data"""
        if category:
            return [item for item in self.training_data if item["category"] == category]
        return self.training_data

legal_db = LegalDatabase()

# -----------------------------------------------------------------------------
# Data loading and NLP
# -----------------------------------------------------------------------------
def clean_html(text):
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"<.*?>", "", text)
    return unescape(clean).strip()

def load_data():
    print("Loading dataset...")
    start = time.time()
    base_dir = os.path.join(os.path.dirname(__file__), "legal_datasets")
    path = os.path.join(base_dir, "justice.csv")
    
    # Load both static data and recent cases
    cases = []
    
    # Load static dataset if available
    if os.path.isfile(path):
        df = pd.read_csv(path)
        # Normalize column names if needed
        if "second_p" in df.columns:
            df = df.rename(columns={"second_p": "second_party"})
        if "decision_t" in df.columns:
            df = df.rename(columns={"decision_t": "decision_type"})
        if "dispositio" in df.columns:
            df = df.rename(columns={"dispositio": "disposition"})
        df["clean_facts"] = df.get("facts", "").fillna("").apply(clean_html)
        combined = []
        for _, row in df.iterrows():
            parts = []
            for col in ["name", "first_party", "second_party", "disposition", "clean_facts"]:
                if col in df.columns and pd.notna(row[col]):
                    parts.append(str(row[col]))
            combined.append(" ".join(parts))
        df["combined"] = combined
        cases = df.to_dict('records')
        print(f"Loaded {len(cases)} static cases")
    else:
        print("Static dataset not found, using live sources only")
        cases = []
    
    print(f"Data loading completed in {time.time()-start:.2f} sec")
    return cases

def setup_nlp():
    print("Loading spaCy model...")
    start = time.time()
    try:
        nlp_ = spacy.load("en_core_web_md", disable=["parser", "ner"])
    except Exception:
        nlp_ = spacy.blank("en")
        print("Using basic English model - consider installing en_core_web_md for better results")
    print(f"spaCy model loaded in {time.time()-start:.2f} sec")
    return nlp_

df_cases = load_data()
nlp = setup_nlp()

def create_search_index(docs):
    print("Building TF-IDF index...")
    start = time.time()
    if not docs:
        return None, None
    
    vectorizer = TfidfVectorizer(max_features=800, stop_words="english", min_df=2, max_df=0.8)
    tfidf_matrix = vectorizer.fit_transform(docs)
    print(f"TF-IDF index done in {time.time()-start:.2f} sec")
    return vectorizer, tfidf_matrix

# Only create index if we have static data
if df_cases and len(df_cases) > 0:
    combined_texts = [case.get('combined', '') for case in df_cases]
    vectorizer, tfidf_matrix = create_search_index(combined_texts)
else:
    vectorizer, tfidf_matrix = None, None

# -----------------------------------------------------------------------------
# Enhanced Training Functions
# -----------------------------------------------------------------------------
def train_legal_knowledge(questions: List[str], answers: List[str]):
    """Train the system with legal knowledge base"""
    trained_count = 0
    for question, answer in zip(questions, answers):
        legal_db.add_training_data(question, answer, "legal_knowledge")
        trained_count += 1
    return trained_count

def generate_training_embeddings():
    """Generate embeddings for training data to improve search"""
    if not legal_db.training_data:
        return
    
    training_texts = [f"{item['question']} {item['answer']}" for item in legal_db.training_data]
    # Store embeddings for semantic search
    # This would be implemented with a proper vector database in production

# -----------------------------------------------------------------------------
# Search Functions
# -----------------------------------------------------------------------------
@lru_cache(maxsize=256)
def search_cases(text, top_k=5, threshold=0.15):
    results = []
    
    # Search live databases first
    live_results = legal_db.search_live_cases(text)
    for result in live_results[:top_k]:
        results.append({
            'type': 'live',
            'data': result,
            'score': 0.9  # High score for live results
        })
    
    # Search static dataset if available
    if vectorizer and tfidf_matrix is not None:
        try:
            vec = vectorizer.transform([text])
            sims = (tfidf_matrix @ vec.T).toarray().flatten()
            idxs = sims.argsort()[::-1]
            for idx in idxs:
                if sims[idx] < threshold or len(results) >= top_k:
                    break
                if idx < len(df_cases):
                    results.append({
                        'type': 'static',
                        'data': df_cases[idx],
                        'score': sims[idx]
                    })
        except Exception as e:
            print(f"Static search error: {e}")
    
    return results[:top_k]

def cosine_sim(vec, mat):
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return mat @ vec

# -----------------------------------------------------------------------------
# In-memory document store
# -----------------------------------------------------------------------------
document_store = {}

def store_document(text, doc_type="general"):
    doc_id = str(uuid.uuid4())
    document_store[doc_id] = {
        "text": text[:5000],
        "type": doc_type,
        "timestamp": datetime.now().isoformat()
    }
    return doc_id

# -----------------------------------------------------------------------------
# Enhanced OpenAI Integration with Training
# -----------------------------------------------------------------------------
def build_enhanced_messages(user_msg: str, history: list):
    """Build messages with enhanced context from training data"""
    
    # System prompt with enhanced knowledge - REMOVED MARKDOWN FORMATTING
    system_prompt = (
        "You are JustiBot, a senior Indian legal research assistant with specialized knowledge in Indian law. "
        "You have access to comprehensive legal databases and training data. "
        "Answer concisely with accurate points and include citations to relevant statutes and case law when appropriate. "
        "If information is insufficient, ask for missing facts. Avoid fabricating citations. "
        "Always be transparent about the limitations of your knowledge regarding recent developments. "
        "Important: My knowledge is based on training data up to October 2023. "
        "For the most current legal developments, consult official legal databases or practicing legal professionals."
        "DO NOT use markdown formatting like **bold** or *italic* in your responses. Use plain text only."
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add relevant training data context
    relevant_training = find_relevant_training(user_msg)
    if relevant_training:
        training_context = "\n\nRelevant legal knowledge:\n" + "\n".join(
            [f"Q: {item['question']}\nA: {item['answer']}" for item in relevant_training[:3]]
        )
        messages[0]["content"] += training_context
    
    # Add conversation history
    if isinstance(history, list):
        for t in history:
            role = t.get("role")
            content = t.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                # Clean any markdown from previous responses
                cleaned_content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove **bold**
                cleaned_content = re.sub(r'\*(.*?)\*', r'\1', cleaned_content)  # Remove *italic*
                messages.append({"role": role, "content": cleaned_content})
    
    messages.append({"role": "user", "content": user_msg})
    return messages

def find_relevant_training(query: str, max_results: int = 3):
    """Find relevant training data based on query"""
    if not legal_db.training_data:
        return []
    
    # Simple keyword-based matching (could be enhanced with embeddings)
    query_lower = query.lower()
    relevant = []
    
    for item in legal_db.training_data:
        question = item['question'].lower()
        answer = item['answer'].lower()
        
        # Check if query terms appear in question or answer
        score = sum(1 for word in query_lower.split() if word in question or word in answer)
        if score > 0:
            relevant.append((item, score))
    
    # Sort by relevance score and return top results
    relevant.sort(key=lambda x: x[1], reverse=True)
    return [item[0] for item in relevant[:max_results]]

def clean_response(text: str) -> str:
    """Clean markdown formatting from response text"""
    if not text:
        return text
    
    # Remove **bold** formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    # Remove *italic* formatting
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    # Remove other markdown elements
    text = re.sub(r'#+\s*', '', text)  # Remove headers
    text = re.sub(r'`(.*?)`', r'\1', text)  # Remove inline code
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Remove links but keep text
    
    return text

# -----------------------------------------------------------------------------
# Routes - Static and Health
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health():
    return jsonify({
        "ok": True, 
        "rows": len(df_cases), 
        "ready": True, 
        "documents_stored": len(document_store),
        "training_examples": len(legal_db.training_data)
    })

@app.route("/stats")
def stats():
    return jsonify({
        "cases": len(df_cases), 
        "documents": len(document_store),
        "training_data": len(legal_db.training_data)
    })

@app.route("/api-status")
def api_status():
    uptime = time.time() - app.config.get("START_TIME", time.time())
    return jsonify({
        "status": "active", 
        "uptime": uptime, 
        "cases": len(df_cases), 
        "documents": len(document_store),
        "training_examples": len(legal_db.training_data)
    })

@app.route('/logo.png')
def serve_logo():
    return send_from_directory('.', 'logo.png')

# -----------------------------------------------------------------------------
# Document Management Routes
# -----------------------------------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json() or {}
        text = data.get("text")
        t = data.get("type", "general")
        if not text:
            return jsonify({"error": "No text provided"}), 400
        doc_id = store_document(text, t)
        return jsonify({"success": True, "id": doc_id, "message": "Document stored."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/documents", methods=["GET"])
def documents():
    return jsonify({
        "count": len(document_store),
        "documents": [
            {"id": k, "type": v["type"], "size": len(v["text"]), "timestamp": v["timestamp"]}
            for k, v in document_store.items()
        ],
    })

@app.route("/clear-documents", methods=["POST"])
def clear_docs():
    global document_store
    document_store = {}
    return jsonify({"success": True, "message": "Cleared all documents."})

# -----------------------------------------------------------------------------
# Training Routes
# -----------------------------------------------------------------------------
@app.route("/train", methods=["POST"])
def train_model():
    try:
        data = request.get_json() or {}
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        category = data.get("category", "general")
        
        if len(questions) != len(answers):
            return jsonify({"error": "Questions and answers must be of equal length"}), 400
        
        trained_count = 0
        for question, answer in zip(questions, answers):
            legal_db.add_training_data(question, answer, category)
            trained_count += 1
        
        generate_training_embeddings()
        
        return jsonify({
            "success": True, 
            "trained_count": trained_count,
            "total_training_examples": len(legal_db.training_data)
        })
    except Exception as e:
        return jsonify({"error": "training_error", "message": str(e)}), 500

@app.route("/training-data", methods=["GET"])
def get_training_data():
    category = request.args.get("category")
    data = legal_db.get_training_data(category)
    return jsonify({
        "count": len(data),
        "training_data": data
    })

# -----------------------------------------------------------------------------
# Search and Case Routes
# -----------------------------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.get_json() or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "No search query provided"}), 400
        
        results = search_cases(query)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": "search_error", "message": str(e)}), 500

# -----------------------------------------------------------------------------
# Enhanced OpenAI Integrated Chat
# -----------------------------------------------------------------------------
def params_from_payload(data: dict):
    model = data.get("model") or "gpt-4o-mini"
    temperature = float(data.get("temperature") or 0.2)
    max_tokens = int(data.get("max_tokens") or 900)
    return model, temperature, max_tokens

@app.route("/openai/chat", methods=["POST"])
def openai_chat():
    try:
        data = request.get_json(silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not user_msg:
            return jsonify({"response": "Please provide a prompt."}), 200

        model, temperature, max_tokens = params_from_payload(data)
        messages = build_enhanced_messages(user_msg, history)

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        answer = resp.choices[0].message.content.strip()
        # Clean the response to remove markdown formatting
        cleaned_answer = clean_response(answer)
        return jsonify({"response": cleaned_answer})
    except Exception as e:
        return jsonify({"error": "openai_error", "message": str(e)}), 500

@app.route("/openai/chat-stream", methods=["POST"])
def openai_chat_stream():
    try:
        data = request.get_json(silent=True) or {}
        user_msg = (data.get("message") or "").strip()
        history = data.get("history") or []
        if not user_msg:
            return Response("data: Please provide a prompt.\n\ndata: [DONE]\n\n", mimetype="text/event-stream")

        model, temperature, max_tokens = params_from_payload(data)
        messages = build_enhanced_messages(user_msg, history)

        def token_stream():
            try:
                stream = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=messages,
                    stream=True,
                )
                accumulated_text = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        content = delta.content
                        accumulated_text += content
                        # Clean the content in real-time
                        cleaned_content = clean_response(content)
                        yield f"data: {cleaned_content}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: [ERROR] {str(e)}\n\n"

        return Response(stream_with_context(token_stream()), mimetype="text/event-stream")
    except Exception as e:
        return Response(f"data: [ERROR] {str(e)}\n\n", mimetype="text/event-stream")

# -----------------------------------------------------------------------------
# Pre-load with some legal training data
# -----------------------------------------------------------------------------
def preload_training_data():
    """Pre-load with some basic legal training data"""
    basic_legal_qa = [
        {
            "question": "What is Article 21 of the Indian Constitution?",
            "answer": "Article 21 of the Indian Constitution states: 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' It has been interpreted broadly by the Supreme Court to include right to livelihood, clean environment, health, and dignity."
        },
        {
            "question": "What is the difference between civil law and criminal law?",
            "answer": "Civil law deals with disputes between individuals/organizations where compensation may be awarded to the victim. Criminal law deals with crimes against the state/society where punishment is imposed. Civil cases are filed by private parties, while criminal cases are filed by the government."
        },
        {
            "question": "What is the Limitation Act in India?",
            "answer": "The Limitation Act, 1963 prescribes time limits for different types of legal actions. For example: contract disputes - 3 years, recovery of debt - 3 years, suits relating to immovable property - 12 years. After the limitation period expires, the right to initiate legal action is extinguished."
        }
    ]
    
    for qa in basic_legal_qa:
        legal_db.add_training_data(qa["question"], qa["answer"], "constitutional_law")
    
    print(f"Pre-loaded {len(basic_legal_qa)} training examples")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Pre-load some training data
    preload_training_data()
    
    app.config["START_TIME"] = time.time()
    print("JustiBot AI Legal Assistant starting...")
    print(f"OpenAI API Key: {'Loaded' if OPENAI_API_KEY else 'Missing'}")
    print(f"Loaded {len(df_cases)} legal cases")
    print(f"Pre-loaded {len(legal_db.training_data)} training examples")
    
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)