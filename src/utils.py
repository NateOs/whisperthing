import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "call_analyzer.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set third-party loggers to WARNING to reduce noise
    logging.getLogger("whisper").setLevel(logging.WARNING)
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

def validate_audio_file(file_path: Path) -> bool:
    """Validate if the audio file is suitable for processing."""
    if not file_path.exists():
        return False
    
    if file_path.suffix.lower() != '.wav':
        return False
    
    # Check file size (max 100MB by default)
    max_size = 100 * 1024 * 1024  # 100MB
    if file_path.stat().st_size > max_size:
        return False
    
    return True

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS format."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"
