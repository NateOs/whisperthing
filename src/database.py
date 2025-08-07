import logging
import pyodbc
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import asdict
from .config import DatabaseConfig
from .models import CallRecord, TranscriptionResult, AnalysisResult

class DatabaseManager:
    """Manages database connections and operations for call analysis data."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.connection_string = self._build_connection_string()
        self._initialize_database()
    
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string."""
        if self.config.trusted_connection:
            return (
                f"DRIVER={{{self.config.driver}}};"
                f"SERVER={self.config.server};"
                f"DATABASE={self.config.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            return (
                f"DRIVER={{{self.config.driver}}};"
                f"SERVER={self.config.server};"
                f"DATABASE={self.config.database};"
                f"UID={self.config.username};"
                f"PWD={self.config.password};"
            )
    
    def _get_connection(self):
        """Get database connection."""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def _initialize_database(self):
        """Initialize database tables if they don't exist."""
        create_tables_sql = """
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='calls' AND xtype='U')
        CREATE TABLE calls (
            id INT IDENTITY(1,1) PRIMARY KEY,
            audio_file_path NVARCHAR(500) NOT NULL,
            processed_date DATETIME2 DEFAULT GETDATE(),
            duration_seconds FLOAT,
            file_size_bytes BIGINT,
            status NVARCHAR(50) DEFAULT 'completed'
        );
        
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='transcriptions' AND xtype='U')
        CREATE TABLE transcriptions (
            id INT IDENTITY(1,1) PRIMARY KEY,
            call_id INT FOREIGN KEY REFERENCES calls(id),
            speaker_type NVARCHAR(20) NOT NULL, -- 'agent' or 'customer'
            text NVARCHAR(MAX) NOT NULL,
            start_time FLOAT NOT NULL,
            end_time FLOAT NOT NULL,
            confidence FLOAT
        );
        
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='keyword_detections' AND xtype='U')
        CREATE TABLE keyword_detections (
            id INT IDENTITY(1,1) PRIMARY KEY,
            call_id INT FOREIGN KEY REFERENCES calls(id),
            keyword NVARCHAR(100) NOT NULL,
            speaker_type NVARCHAR(20) NOT NULL,
            timestamp_seconds FLOAT NOT NULL,
            context_text NVARCHAR(500),
            confidence FLOAT
        );
        
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='call_metrics' AND xtype='U')
        CREATE TABLE call_metrics (
            id INT IDENTITY(1,1) PRIMARY KEY,
            call_id INT FOREIGN KEY REFERENCES calls(id),
            total_agent_talk_time FLOAT,
            total_customer_talk_time FLOAT,
            agent_word_count INT,
            customer_word_count INT,
            total_keywords_found INT,
            politeness_score FLOAT
        );
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_tables_sql)
                conn.commit()
                self.logger.info("Database tables initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
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
    
    def get_call_by_id(self, call_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve call data by ID."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT c.*, cm.* FROM calls c
                    LEFT JOIN call_metrics cm ON c.id = cm.call_id
                    WHERE c.id = ?
                """, (call_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(zip([column[0] for column in cursor.description], result))
                return None
                
        except Exception as e:
            self.logger.error(f"Error retrieving call {call_id}: {str(e)}")
            raise