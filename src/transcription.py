import logging
import time
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import openai
from .audio_utils import preprocess_audio
import tempfile
import soundfile as sf
from pyannote.audio import Pipeline
from pydub import AudioSegment

class SimpleTranscriptionService:
    """Simple transcription service using OpenAI API and pyannote for diarization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = openai.OpenAI()
        self.diarization_pipeline = None
        self._load_diarization_model()
    
    def _load_diarization_model(self):
        """Load the speaker diarization model."""
        try:
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token and hf_token != "your_huggingface_token_here":
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                self.logger.info("Speaker diarization model loaded successfully")
            else:
                self.logger.warning("No valid HuggingFace token provided, diarization disabled")
        except Exception as e:
            self.logger.error(f"Failed to load diarization model: {e}")
    
    def transcribe_audio(self, audio_file_path: Path) -> Dict[str, Any]:
        """Transcribe audio file and perform speaker diarization."""
        try:
            self.logger.info(f"Processing audio file: {audio_file_path}")
            
            # Step 1: Transcribe with OpenAI
            transcription = self._transcribe_with_openai(audio_file_path)
            
            # Step 2: Perform speaker diarization
            speaker_segments = self._perform_diarization(audio_file_path)
            
            # Step 3: Combine results
            result = {
                "file_path": str(audio_file_path),
                "transcription": transcription,
                "speaker_segments": speaker_segments,
                "combined_result": self._combine_transcription_and_diarization(
                    transcription, speaker_segments
                )
            }
            
            self.logger.info("Audio processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing audio file {audio_file_path}: {e}")
            raise
    
    def _transcribe_with_openai(self, audio_file_path: Path) -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper API."""
        try:
            # Preprocess audio
            audio_data, sr = preprocess_audio(audio_file_path)
            
            # Save preprocessed audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, audio_data, sr)
                
                # Transcribe with better parameters
                with open(temp_file.name, 'rb') as audio:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio,
                        language="es",  # Specify Spanish
                        prompt="Esta es una llamada de servicio al cliente en español. Los participantes discuten sobre pedidos, direcciones y códigos postales.",  # Context prompt
                        temperature=0.0,  # More deterministic
                        response_format="verbose_json"
                    )
            
            return self._process_response(response, audio_file_path)
            
        except Exception as e:
            self.logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    def _perform_diarization(self, audio_file_path: Path) -> List[Dict[str, Any]]:
        """Perform speaker diarization."""
        if not self.diarization_pipeline:
            self.logger.warning("Diarization pipeline not available")
            return []
        
        try:
            diarization = self.diarization_pipeline(str(audio_file_path))
            
            speaker_segments = []
            speaker_times = {}
            
            # Collect all speaker segments and calculate total speaking time
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                }
                speaker_segments.append(segment)
                
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += segment["duration"]
            
            # Assign roles based on speaking time (most talkative = agent)
            if len(speaker_times) >= 2:
                sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
                speaker_mapping = {
                    sorted_speakers[0][0]: "agent",
                    sorted_speakers[1][0]: "customer"
                }
                
                # Add additional speakers as "other" if present
                for i, (speaker, _) in enumerate(sorted_speakers[2:], start=2):
                    speaker_mapping[speaker] = f"speaker_{i+1}"
            else:
                # Single speaker or no speakers detected
                speaker_mapping = {list(speaker_times.keys())[0]: "agent"} if speaker_times else {}
            
            # Apply role mapping
            for segment in speaker_segments:
                segment["role"] = speaker_mapping.get(segment["speaker"], "unknown")
            
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return []
    
    def _combine_transcription_and_diarization(self, transcription: Dict[str, Any], 
                                             speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine transcription words with speaker information."""
        if not transcription.get("words") or not speaker_segments:
            return []
        
        combined = []
        
        for word_info in transcription["words"]:
            word_start = word_info.get("start", 0)
            word_end = word_info.get("end", 0)
            
            # Find which speaker segment this word belongs to
            speaker_role = "unknown"
            for segment in speaker_segments:
                if segment["start"] <= word_start <= segment["end"]:
                    speaker_role = segment["role"]
                    break
            
            combined.append({
                "word": word_info.get("word", ""),
                "start": word_start,
                "end": word_end,
                "speaker": speaker_role
            })
        
        return combined
    
    def _process_response(self, response, audio_file):
        """Process and clean the transcription response."""
        # Filter out repetitive segments
        segments = []
        prev_text = ""
        repetition_count = 0
        
        for segment in response.segments:
            text = segment['text'].strip()
            
            # Skip empty or very short segments
            if len(text) < 3:
                continue
                
            # Check for repetitive content
            if text == prev_text:
                repetition_count += 1
                if repetition_count > 2:  # Skip after 2 repetitions
                    continue
            else:
                repetition_count = 0
                
            # Skip segments that are just "Mami" repeated
            if text.lower().strip() in ['mami', 'mami.']:
                continue
                
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': text,
                'confidence': getattr(segment, 'confidence', 0.8)
            })
            
            prev_text = text
        
        return {
            'text': response.text,
            'segments': segments,
            'duration': getattr(response, 'duration', 0),
            'language': 'es'
        }