# backup_database.py
import sqlite3
import shutil
import datetime
import os

def backup_database():
    """Create a backup of the database"""
    if not os.path.exists('legal_assistant.db'):
        print("No database found to backup")
        return
    
    # Create backups directory if it doesn't exist
    if not os.path.exists('backups'):
        os.makedirs('backups')
    
    # Create backup filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = f'backups/legal_assistant_backup_{timestamp}.db'
    
    # Copy the database file
    shutil.copy2('legal_assistant.db', backup_file)
    print(f"Database backed up to: {backup_file}")
    
    # Clean up old backups (keep last 7 days)
    cleanup_old_backups()

def cleanup_old_backups(days_to_keep=7):
    """Remove backup files older than specified days"""
    if not os.path.exists('backups'):
        return
    
    cutoff_time = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
    
    for filename in os.listdir('backups'):
        if filename.startswith('legal_assistant_backup_') and filename.endswith('.db'):
            file_path = os.path.join('backups', filename)
            file_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            
            if file_time < cutoff_time:
                os.remove(file_path)
                print(f"Removed old backup: {filename}")

if __name__ == "__main__":
    backup_database()