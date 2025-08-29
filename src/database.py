import pyodbc
import logging
import os
from typing import Optional, Dict, Any, List
from .config import DatabaseConfig
from .models import CallRecord, TranscriptionResult, AnalysisResult

class DatabaseManager:
    """Manages database connections and operations for call analysis."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Check if we have a custom connection string from environment
        custom_connection_string = os.getenv("SQL_CONNECTION_STRING")
        if custom_connection_string:
            self.connection_string = custom_connection_string
        else:
            # Build connection string with SSL bypass for localhost
            if config.trusted_connection:
                self.connection_string = (
                    f"DRIVER={{{config.driver}}};"
                    f"SERVER={config.server};"
                    f"DATABASE={config.database};"
                    f"Trusted_Connection=yes;"
                    f"TrustServerCertificate=yes;"
                    f"Encrypt=optional;"
                )
            else:
                self.connection_string = (
                    f"DRIVER={{{config.driver}}};"
                    f"SERVER={config.server};"
                    f"DATABASE={config.database};"
                    f"UID={config.username};"
                    f"PWD={config.password};"
                    f"TrustServerCertificate=yes;"
                    f"Encrypt=optional;"
                )
        
        self.logger.info(f"Database connection string configured for {config.server}/{config.database}")
    
    def _get_connection(self):
        """Get database connection."""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection and return True if successful."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result and result[0] == 1:
                    self.logger.info("Database connection test successful")
                    return True
                else:
                    self.logger.error("Database connection test failed - unexpected result")
                    return False
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def get_database_info(self) -> Optional[Dict[str, Any]]:
        """Get basic database information."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get database name
                cursor.execute("SELECT DB_NAME()")
                db_name = cursor.fetchone()[0]
                
                # Get table count
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_TYPE = 'BASE TABLE'
                """)
                table_count = cursor.fetchone()[0]
                
                # Get SQL Server version
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0]
                
                return {
                    "database_name": db_name,
                    "table_count": table_count,
                    "version": version
                }
                
        except Exception as e:
            self.logger.error(f"Error getting database info: {str(e)}")
            return None
    
    def save_call_analysis(self, audio_file_path: str, transcription: TranscriptionResult, 
                          analysis: AnalysisResult) -> int:
        """Save complete call analysis to database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert call record
                cursor.execute("""
                    INSERT INTO calls (audio_file_path, duration_seconds, file_size_bytes)
                    OUTPUT INSERTED.id
                    VALUES (?, ?, ?)
                """, (audio_file_path, transcription.duration, transcription.file_size))
                
                call_id = cursor.fetchone()[0]
                
                # Insert transcription segments
                for segment in transcription.segments:
                    cursor.execute("""
                        INSERT INTO transcriptions (call_id, speaker_type, text, start_time, end_time, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (call_id, segment.speaker, segment.text, segment.start, segment.end, segment.confidence))
                
                # Insert keyword detections
                for detection in analysis.keyword_detections:
                    cursor.execute("""
                        INSERT INTO keyword_detections (call_id, keyword, speaker_type, timestamp_seconds, context_text, confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (call_id, detection.keyword, detection.speaker, detection.timestamp, 
                          detection.context, detection.confidence))
                
                # Insert call metrics
                metrics = analysis.metrics
                cursor.execute("""
                    INSERT INTO call_metrics (call_id, total_agent_talk_time, total_customer_talk_time,
                                            agent_word_count, customer_word_count, total_keywords_found, politeness_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (call_id, metrics.agent_talk_time, metrics.customer_talk_time,
                      metrics.agent_word_count, metrics.customer_word_count,
                      metrics.total_keywords_found, metrics.politeness_score))
                
                conn.commit()
                self.logger.info(f"Call analysis saved successfully with ID: {call_id}")
                return call_id
                
        except Exception as e:
            self.logger.error(f"Error saving call analysis: {str(e)}")
            raise