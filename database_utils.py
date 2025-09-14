# database_utils.py (optional - can be added to app.py)
import sqlite3
from contextlib import contextmanager

@contextmanager
def db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('legal_assistant.db')
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_chat_history(session_id):
    """Retrieve chat history for a session"""
    with db_connection() as conn:
        cursor = conn.execute(
            "SELECT history FROM chat_sessions WHERE id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        return result['history'] if result else None

def save_chat_history(session_id, history):
    """Save chat history for a session"""
    with db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chat_sessions (id, history, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
            (session_id, history)
        )
        conn.commit()

def get_uploaded_documents(session_id):
    """Get all documents uploaded for a session"""
    with db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, filename, file_type, created_at FROM uploaded_documents WHERE session_id = ? ORDER BY created_at DESC",
            (session_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

def cleanup_old_sessions(days_to_keep=30):
    """Clean up old chat sessions and their documents"""
    with db_connection() as conn:
        # Delete sessions older than X days
        conn.execute(
            "DELETE FROM chat_sessions WHERE created_at < datetime('now', ?)",
            (f'-{days_to_keep} days',)
        )
        
        # Delete orphaned documents
        conn.execute(
            "DELETE FROM uploaded_documents WHERE session_id NOT IN (SELECT id FROM chat_sessions)"
        )
        
        conn.commit()