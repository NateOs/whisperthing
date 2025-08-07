import logging
import time
import os
from pathlib import Path
from typing import List, Tuple
import whisper
import torch
from pydub import AudioSegment
from pyannote.audio import Pipeline
from .config import WhisperConfig
from .models import TranscriptionSegment, TranscriptionResult

class TranscriptionService:
    """Handles audio transcription using Whisper and speaker diarization."""
    
    def __init__(self, config: WhisperConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and speaker diarization models."""
        try:
            # Load Whisper model
            self.logger.info(f"Loading Whisper model: {self.config.model_size}")
            self.whisper_model = whisper.load_model(
                self.config.model_size,
                device=self.config.device
            )
            
            # Load speaker diarization model (requires HuggingFace token)
            # You'll need to set HUGGINGFACE_TOKEN environment variable
            try:
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
                )
                self.diarization_available = True
                self.logger.info("Speaker diarization model loaded successfully")
            except Exception as e:
                self.logger.warning(f"Speaker diarization not available: {str(e)}")
                self.diarization_available = False
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def transcribe_call(self, audio_file_path: Path) -> TranscriptionResult:
        """Transcribe a call recording with speaker separation."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting transcription of {audio_file_path}")
            
            # Get file info
            file_size = audio_file_path.stat().st_size
            
            # Load and preprocess audio
            audio = AudioSegment.from_wav(str(audio_file_path))
            duration = len(audio) / 1000.0  # Convert to seconds
            
            # Perform speaker diarization if available
            speaker_segments = []
            if self.diarization_available:
                speaker_segments = self._perform_diarization(audio_file_path)
            
            # Transcribe audio
            result = self.whisper_model.transcribe(
                str(audio_file_path),
                language=self.config.language,
                word_timestamps=True
            )
            
            # Process segments and assign speakers
            segments = self._process_segments(result, speaker_segments, duration)
            
            processing_time = time.time() - start_time
            
            transcription_result = TranscriptionResult(
                segments=segments,
                duration=duration,
                file_size=file_size,
                language=self.config.language,
                processing_time=processing_time
            )
            
            self.logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            return transcription_result
            
        except Exception as e:
            self.logger.error(f"Error transcribing {audio_file_path}: {str(e)}")
            raise
    
    def _perform_diarization(self, audio_file_path: Path) -> List[Tuple[float, float, str]]:
        """Perform speaker diarization to identify different speakers."""
        try:
            diarization = self.diarization_pipeline(str(audio_file_path))
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map speakers to agent/customer based on speaking time
                # Assumption: speaker with more talk time is likely the agent
                speaker_segments.append((turn.start, turn.end, speaker))
            
            # Determine which speaker is agent vs customer
            speaker_times = {}
            for start, end, speaker in speaker_segments:
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += (end - start)
            
            # Speaker with more time is assumed to be agent
            sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
            speaker_mapping = {}
            if len(sorted_speakers) >= 2:
                speaker_mapping[sorted_speakers[0][0]] = "agent"
                speaker_mapping[sorted_speakers[1][0]] = "customer"
            elif len(sorted_speakers) == 1:
                speaker_mapping[sorted_speakers[0][0]] = "agent"
            
            # Map speakers to agent/customer
            mapped_segments = []
            for start, end, speaker in speaker_segments:
                mapped_speaker = speaker_mapping.get(speaker, "unknown")
                mapped_segments.append((start, end, mapped_speaker))
            
            return mapped_segments
            
        except Exception as e:
            self.logger.warning(f"Diarization failed: {str(e)}")
            return []
    
    def _process_segments(self, whisper_result: dict, speaker_segments: List[Tuple[float, float, str]], 
                         duration: float) -> List[TranscriptionSegment]:
        """Process Whisper segments and assign speakers."""
        segments = []
        
        for segment in whisper_result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            confidence = segment.get("avg_logprob", 0.0)
            
            # Determine speaker for this segment
            speaker = self._determine_speaker(start_time, end_time, speaker_segments)
            
            segments.append(TranscriptionSegment(
                text=text,
                start=start_time,
                end=end_time,
                speaker=speaker,
                confidence=confidence
            ))
        
        return segments
    
    def _determine_speaker(self, start_time: float, end_time: float, 
                          speaker_segments: List[Tuple[float, float, str]]) -> str:
        """Determine speaker for a given time segment."""
        if not speaker_segments:
            return "unknown"
        
        # Find the speaker segment with maximum overlap
        max_overlap = 0
        best_speaker = "unknown"
        
        for seg_start, seg_end, speaker in speaker_segments:
            overlap_start = max(start_time, seg_start)
            overlap_end = min(end_time, seg_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = speaker
        
        return best_speaker