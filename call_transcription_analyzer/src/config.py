import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DatabaseConfig:
    server: str
    database: str
    username: str
    password: str
    driver: str = "ODBC Driver 17 for SQL Server"
    trusted_connection: bool = False

@dataclass
class WhisperConfig:
    model_size: str = "base"
    language: str = "es"
    device: str = "cpu"
    compute_type: str = "int8"

@dataclass
class AnalysisConfig:
    keywords: List[str]
    speaker_separation_enabled: bool = True
    confidence_threshold: float = 0.7

@dataclass
class ProcessingConfig:
    input_directory: str
    completed_directory: str
    move_completed: bool = True
    max_file_size_mb: int = 100

class Config:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            if not self.config_path.exists():
                self._create_default_config()
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            self.database = DatabaseConfig(**config_data['database'])
            self.whisper = WhisperConfig(**config_data['whisper'])
            self.analysis = AnalysisConfig(**config_data['analysis'])
            self.processing = ProcessingConfig(**config_data['processing'])
            
            self.logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _create_default_config(self):
        """Create a default configuration file."""
        default_config = {
            "database": {
                "server": "localhost",
                "database": "CallAnalysis",
                "username": "your_username",
                "password": "your_password",
                "driver": "ODBC Driver 17 for SQL Server",
                "trusted_connection": False
            },
            "whisper": {
                "model_size": "base",
                "language": "es",
                "device": "cpu",
                "compute_type": "int8"
            },
            "analysis": {
                "keywords": [
                    "gracias",
                    "muchas gracias",
                    "de nada",
                    "por favor",
                    "disculpe",
                    "perdón",
                    "buenos días",
                    "buenas tardes",
                    "buenas noches"
                ],
                "speaker_separation_enabled": True,
                "confidence_threshold": 0.7
            },
            "processing": {
                "input_directory": "./input",
                "completed_directory": "./completed",
                "move_completed": True,
                "max_file_size_mb": 100
            }
        }
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Default configuration created at {self.config_path}")
