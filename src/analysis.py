import os
import logging
import re
import time
from typing import List, Dict, Set
from .config import AnalysisConfig
from .models import TranscriptionResult, AnalysisResult, KeywordDetection, CallMetrics

class CallAnalyzer:
    """Analyzes transcribed calls for keywords and metrics."""
    
    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig.from_env()
        self.logger = logging.getLogger(__name__)
        # Load keywords from environment or config
        self.keywords = self._load_keywords_from_env()
    
    def _load_keywords_from_env(self) -> List[str]:
        """Load keywords from environment variable or use config defaults."""
        # First try to get from environment
        custom_keywords = os.getenv("ANALYSIS_KEYWORDS", "")
        if custom_keywords:
            keywords = [k.strip().lower() for k in custom_keywords.split(",")]
            self.logger.info(f"Loaded {len(keywords)} keywords from environment")
            return keywords
        
        # Fall back to config keywords
        if self.config and self.config.keywords:
            keywords = [kw.lower() for kw in self.config.keywords]
            self.logger.info(f"Using {len(keywords)} keywords from config")
            return keywords
        
        # Default Spanish keywords for call analysis
        default_keywords = [
            "gracias", "muchas gracias", "por favor", "disculpe", "perdón",
            "problema", "ayuda", "servicio", "cliente", "factura", "pago",
            "cancelar", "activar", "desactivar", "consulta", "queja",
            "buenos días", "buenas tardes", "buenas noches", "de nada",
            "con permiso", "si me permite", "entiendo", "perfecto",
            "lo siento", "mil disculpas", "que tenga buen día", "hasta luego"
        ]
        
        self.logger.info(f"Using {len(default_keywords)} default keywords")
        return default_keywords
    
    def scan_transcript_for_keywords(self, transcription: TranscriptionResult) -> List[Dict[str, any]]:
        """
        Scan through the transcript and find predefined keywords with their timestamps.
        Returns a list of keyword detections with detailed information.
        """
        keyword_detections = []
        
        if not transcription.segments:
            self.logger.warning("No segments found in transcription")
            return keyword_detections
        
        self.logger.info(f"Scanning transcript for {len(self.keywords)} keywords across {len(transcription.segments)} segments")
        
        for segment in transcription.segments:
            text_lower = segment.text.lower()
            segment_start = segment.start
            segment_end = segment.end
            segment_duration = segment_end - segment_start
            
            for keyword in self.keywords:
                # Find all occurrences of the keyword using word boundaries
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = list(re.finditer(pattern, text_lower))
                
                for match in matches:
                    # Calculate precise timestamp within the segment
                    char_position = match.start()
                    text_length = len(segment.text)
                    
                    # Estimate timestamp based on character position
                    if text_length > 0:
                        relative_time = (char_position / text_length) * segment_duration
                    else:
                        relative_time = 0
                    
                    timestamp = segment_start + relative_time
                    
                    # Extract context around the keyword
                    context = self._extract_context(segment.text, match.start(), match.end())
                    
                    # Create detection record
                    detection = {
                        "keyword": keyword,
                        "timestamp": round(timestamp, 2),
                        "speaker": getattr(segment, 'speaker', 'unknown'),
                        "context": context,
                        "confidence": getattr(segment, 'confidence', 0.8),
                        "segment_start": segment_start,
                        "segment_end": segment_end,
                        "segment_text": segment.text,
                        "match_start_char": match.start(),
                        "match_end_char": match.end()
                    }
                    
                    keyword_detections.append(detection)
                    
                    self.logger.debug(f"Keyword '{keyword}' found at {timestamp:.2f}s by {detection['speaker']}: '{context}'")
        
        # Sort detections by timestamp
        keyword_detections.sort(key=lambda x: x["timestamp"])
        
        self.logger.info(f"Found {len(keyword_detections)} total keyword occurrences")
        
        # Log summary by keyword
        keyword_counts = {}
        for detection in keyword_detections:
            keyword = detection["keyword"]
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        for keyword, count in sorted(keyword_counts.items()):
            self.logger.info(f"  '{keyword}': {count} occurrences")
        
        return keyword_detections
    
    def get_keywords_by_speaker(self, keyword_detections: List[Dict[str, any]]) -> Dict[str, List[Dict[str, any]]]:
        """Group keyword detections by speaker."""
        by_speaker = {}
        
        for detection in keyword_detections:
            speaker = detection["speaker"]
            if speaker not in by_speaker:
                by_speaker[speaker] = []
            by_speaker[speaker].append(detection)
        
        return by_speaker
    
    def get_keywords_by_timeframe(self, keyword_detections: List[Dict[str, any]], 
                                 timeframe_seconds: float = 60.0) -> List[Dict[str, any]]:
        """Group keyword detections by time frames."""
        timeframes = []
        
        if not keyword_detections:
            return timeframes
        
        current_frame_start = 0
        current_frame_keywords = []
        
        for detection in keyword_detections:
            timestamp = detection["timestamp"]
            
            # Check if we need to start a new timeframe
            if timestamp >= current_frame_start + timeframe_seconds:
                # Save current frame if it has keywords
                if current_frame_keywords:
                    timeframes.append({
                        "start_time": current_frame_start,
                        "end_time": current_frame_start + timeframe_seconds,
                        "keywords": current_frame_keywords.copy(),
                        "keyword_count": len(current_frame_keywords)
                    })
                
                # Start new frame
                current_frame_start = int(timestamp // timeframe_seconds) * timeframe_seconds
                current_frame_keywords = []
            
            current_frame_keywords.append(detection)
        
        # Add the last frame
        if current_frame_keywords:
            timeframes.append({
                "start_time": current_frame_start,
                "end_time": current_frame_start + timeframe_seconds,
                "keywords": current_frame_keywords,
                "keyword_count": len(current_frame_keywords)
            })
        
        return timeframes

    def analyze_call(self, transcription: TranscriptionResult) -> AnalysisResult:
        """Perform complete analysis of a transcribed call."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting call analysis")
            
            # Detect keywords using the enhanced scanner
            keyword_detections_raw = self.scan_transcript_for_keywords(transcription)
            
            # Convert to KeywordDetection objects for compatibility
            keyword_detections = []
            for detection in keyword_detections_raw:
                kd = KeywordDetection(
                    keyword=detection["keyword"],
                    timestamp=detection["timestamp"],
                    speaker=detection["speaker"],
                    context=detection["context"],
                    confidence=detection["confidence"]
                )
                keyword_detections.append(kd)
            
            # Calculate metrics
            metrics = self._calculate_metrics(transcription, keyword_detections)
            
            analysis_time = time.time() - start_time
            
            result = AnalysisResult(
                keyword_detections=keyword_detections,
                metrics=metrics,
                analysis_time=analysis_time
            )
            
            self.logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during call analysis: {str(e)}")
            raise
    
    def _detect_keywords(self, transcription: TranscriptionResult) -> List[KeywordDetection]:
        """Detect configured keywords in the transcription."""
        detections = []
        
        for segment in transcription.segments:
            text_lower = segment.text.lower()
            
            for keyword in self.keywords:
                # Find all occurrences of the keyword
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = list(re.finditer(pattern, text_lower))
                
                for match in matches:
                    # Calculate approximate timestamp within the segment
                    char_position = match.start()
                    segment_duration = segment.end - segment.start
                    text_length = len(segment.text)
                    
                    # Estimate timestamp based on character position
                    relative_time = (char_position / text_length) * segment_duration if text_length > 0 else 0
                    timestamp = segment.start + relative_time
                    
                    # Extract context (surrounding words)
                    context = self._extract_context(segment.text, match.start(), match.end())
                    
                    detection = KeywordDetection(
                        keyword=keyword,
                        timestamp=timestamp,
                        speaker=segment.speaker,
                        context=context,
                        confidence=segment.confidence
                    )
                    
                    detections.append(detection)
                    
                    self.logger.debug(f"Keyword '{keyword}' detected at {timestamp:.2f}s by {segment.speaker}")
        
        return detections
    
    def _extract_context(self, text: str, start_pos: int, end_pos: int, context_words: int = 5) -> str:
        """Extract context around a keyword match."""
        words = text.split()
        
        # Find word positions
        char_count = 0
        start_word_idx = 0
        end_word_idx = len(words)
        
        for i, word in enumerate(words):
            if char_count <= start_pos < char_count + len(word):
                start_word_idx = i
            if char_count <= end_pos <= char_count + len(word):
                end_word_idx = i + 1
                break
            char_count += len(word) + 1  # +1 for space
        
        # Extract context
        context_start = max(0, start_word_idx - context_words)
        context_end = min(len(words), end_word_idx + context_words)
        
        context_words_list = words[context_start:context_end]
        return ' '.join(context_words_list)
    
    def _calculate_metrics(self, transcription: TranscriptionResult, 
                          detections: List[KeywordDetection]) -> CallMetrics:
        """Calculate various metrics from the transcription."""
        agent_talk_time = 0.0
        customer_talk_time = 0.0
        agent_word_count = 0
        customer_word_count = 0
        
        # Calculate talk times and word counts
        for segment in transcription.segments:
            duration = segment.end - segment.start
            word_count = len(segment.text.split())
            
            if segment.speaker == "agent":
                agent_talk_time += duration
                agent_word_count += word_count
            elif segment.speaker == "customer":
                customer_talk_time += duration
                customer_word_count += word_count
        
        # Calculate politeness score based on polite keywords
        polite_keywords = {
            "gracias", "muchas gracias", "de nada", "por favor", 
            "disculpe", "perdón", "buenos días", "buenas tardes", 
            "buenas noches", "con permiso", "si me permite"
        }
        
        polite_detections = [d for d in detections if d.keyword in polite_keywords]
        total_segments = len(transcription.segments)
        politeness_score = len(polite_detections) / max(total_segments, 1) * 100
        
        return CallMetrics(
            agent_talk_time=agent_talk_time,
            customer_talk_time=customer_talk_time,
            agent_word_count=agent_word_count,
            customer_word_count=customer_word_count,
            total_keywords_found=len(detections),
            politeness_score=min(politeness_score, 100.0)  # Cap at 100%
        )