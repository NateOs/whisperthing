from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed audio."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds
    speaker: str  # 'agent' or 'customer'
    confidence: float

@dataclass
class TranscriptionResult:
    """Complete transcription result for a call."""
    segments: List[TranscriptionSegment]
    text: str  # Full text of the transcription
    duration: float  # Total duration in seconds
    file_size: int   # File size in bytes
    language: str
    processing_time: float = 0.0  # Time taken to process the transcription

@dataclass
class KeywordDetection:
    """Represents a detected keyword in the call."""
    keyword: str
    timestamp: float  # Time in seconds when keyword was mentioned
    speaker: str      # Who said it: 'agent' or 'customer'
    context: str      # Surrounding text for context
    confidence: float

@dataclass
class CallMetrics:
    """Metrics calculated from the call analysis."""
    agent_talk_time: float
    customer_talk_time: float
    agent_word_count: int
    customer_word_count: int
    total_keywords_found: int
    politeness_score: float  # Score based on politeness keywords found

@dataclass
class AnalysisResult:
    """Complete analysis result for a call."""
    keyword_detections: List[KeywordDetection]
    metrics: CallMetrics
    analysis_time: float

@dataclass
class CallRecord:
    """Database record for a processed call."""
    id: Optional[int]
    audio_file_path: str
    processed_date: datetime
    duration_seconds: float
    file_size_bytes: int
    status: str