import logging
import re
import time
from typing import List, Dict, Set
from .config import AnalysisConfig
from .models import TranscriptionResult, AnalysisResult, KeywordDetection, CallMetrics

class CallAnalyzer:
    """Analyzes transcribed calls for keywords and metrics."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.keywords = [kw.lower() for kw in config.keywords]
    
    def analyze_call(self, transcription: TranscriptionResult) -> AnalysisResult:
        """Perform complete analysis of a transcribed call."""
        start_time = time.time()
        
        try:
            self.logger.info("Starting call analysis")
            
            # Detect keywords
            keyword_detections = self._detect_keywords(transcription)
            
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