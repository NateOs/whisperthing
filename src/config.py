import json
import os
from typing import List
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
import logging

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    server: str
    database: str
    username: str
    password: str
    driver: str = ""
    trusted_connection: bool = False

    @classmethod
    def from_env(cls):
        """Create DatabaseConfig from environment variables."""
        return cls(
            server=os.getenv("DB_SERVER", "localhost"),
            database=os.getenv("DB_DATABASE", "CallAnalysis"),
            username=os.getenv("DB_USERNAME", ""),
            password=os.getenv("DB_PASSWORD", ""),
            driver=os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server"),
            trusted_connection=os.getenv("DB_TRUSTED_CONNECTION", "false").lower() == "true"
        )

@dataclass
class WhisperConfig:
    model_size: str = "base"
    language: str = "es"
    device: str = "cpu"
    compute_type: str = "int8"

    @classmethod
    def from_env(cls):
        """Create WhisperConfig from environment variables."""
        return cls(
            model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            language=os.getenv("WHISPER_LANGUAGE", "es"),
            device=os.getenv("WHISPER_DEVICE", "cpu"),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        )

@dataclass
class AnalysisConfig:
    keywords: List[str]
    speaker_separation_enabled: bool = True
    confidence_threshold: float = 0.7

    @classmethod
    def from_env(cls):
        """Create AnalysisConfig from environment variables."""
        # Default Spanish politeness keywords
        default_keywords = [
            "gracias", "muchas gracias", "de nada", "por favor", "disculpe", 
            "perdón", "buenos días", "buenas tardes", "buenas noches",
            "con permiso", "si me permite", "lo siento", "mil disculpas",
            "que tenga buen día", "que esté bien", "hasta luego", "nos vemos"
        ]
        
        # Allow custom keywords from environment (comma-separated)
        keywords_env = os.getenv("ANALYSIS_KEYWORDS")
        keywords = keywords_env.split(",") if keywords_env else default_keywords
        keywords = [kw.strip() for kw in keywords]  # Clean whitespace
        
        return cls(
            keywords=keywords,
            speaker_separation_enabled=os.getenv("SPEAKER_SEPARATION_ENABLED", "true").lower() == "true",
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
        )

@dataclass
class ProcessingConfig:
    input_directory: str
    completed_directory: str
    move_completed: bool = True
    max_file_size_mb: int = 100

    @classmethod
    def from_env(cls):
        """Create ProcessingConfig from environment variables."""
        return cls(
            input_directory=os.getenv("INPUT_DIRECTORY", "./input"),
            completed_directory=os.getenv("COMPLETED_DIRECTORY", "./completed"),
            move_completed=os.getenv("MOVE_COMPLETED", "true").lower() == "true",
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100"))
        )

@dataclass
class Config:
    database: DatabaseConfig
    whisper: WhisperConfig
    analysis: AnalysisConfig
    processing: ProcessingConfig

    @classmethod
    def from_env(cls):
        """Create complete configuration from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            whisper=WhisperConfig.from_env(),
            analysis=AnalysisConfig.from_env(),
            processing=ProcessingConfig.from_env()
        )

def load_config() -> Config:
    """Load configuration from environment variables."""
    logger = logging.getLogger(__name__)
    
    # Ensure .env file is loaded
    load_dotenv()
    
    # Validate required environment variables
    required_vars = ["DB_USERNAME", "DB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    config = Config.from_env()
    logger.info("Configuration loaded successfully from environment variables")
    return config

class ConfigManager:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables first
        load_dotenv()
        
        if not self.config_path.exists():
            self._create_default_config()
        
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration with environment variable overrides."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Override database config with environment variables if they exist
        if os.getenv("DB_SERVER"):
            config["database"]["server"] = os.getenv("DB_SERVER")
        if os.getenv("DB_DATABASE"):
            config["database"]["database"] = os.getenv("DB_DATABASE")
        if os.getenv("DB_USERNAME"):
            config["database"]["username"] = os.getenv("DB_USERNAME")
        if os.getenv("DB_PASSWORD"):
            config["database"]["password"] = os.getenv("DB_PASSWORD")
        if os.getenv("DB_DRIVER"):
            config["database"]["driver"] = os.getenv("DB_DRIVER")
        if os.getenv("DB_TRUSTED_CONNECTION"):
            config["database"]["trusted_connection"] = os.getenv("DB_TRUSTED_CONNECTION").lower() == "true"
        
        return config

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
