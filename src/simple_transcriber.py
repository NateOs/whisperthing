import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from pyannote.audio import Pipeline
from openai import OpenAI
from dotenv import load_dotenv

class SimpleTranscriber:
    """Simple transcription service using OpenAI Whisper API and pyannote for diarization."""
    
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize diarization pipeline
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token or hf_token == "your_huggingface_token_here":
            self.logger.warning("No HuggingFace token provided, diarization will be disabled")
            self.pipeline = None
        else:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization", 
                    use_auth_token=hf_token
                )
                self.logger.info("Diarization pipeline loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load diarization pipeline: {e}")
                self.pipeline = None
    
    def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio file with diarization and transcription."""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Step 1: Run diarization
        speaker_segments = self._run_diarization(str(audio_path))
        
        # Step 2: Run Whisper transcription
        transcript = self._run_transcription(str(audio_path))
        
        # Step 3: Merge based on timestamps
        merged_result = self._merge_transcription_and_speakers(transcript, speaker_segments)
        
        return {
            "file_path": str(audio_path),
            "speaker_segments": speaker_segments,
            "transcript": transcript,
            "merged_result": merged_result
        }
    
    def _run_diarization(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """Run speaker diarization on audio file."""
        if not self.pipeline:
            self.logger.warning("Diarization pipeline not available")
            return []
        
        try:
            diarization = self.pipeline(audio_file_path)
            
            # Convert diarization output to a list
            speaker_segments = []
            speaker_times = {}
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "duration": turn.end - turn.start
                }
                speaker_segments.append(segment)
                
                # Track total speaking time per speaker
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
                
                # Add role to segments
                for segment in speaker_segments:
                    segment["role"] = speaker_mapping.get(segment["speaker"], "unknown")
            else:
                # Single speaker
                for segment in speaker_segments:
                    segment["role"] = "agent"
            
            self.logger.info(f"Diarization completed: {len(speaker_segments)} segments found")
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return []
    
    def _run_transcription(self, audio_file_path: str) -> Dict[str, Any]:
        """Run Whisper transcription on audio file."""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="es",  # Spanish
                    response_format="verbose_json"
                )
            
            # Convert to dict for easier handling
            result = {
                "text": transcript.text,
                "segments": []
            }
            
            # Extract segments if available
            if hasattr(transcript, 'segments'):
                for segment in transcript.segments:
                    result["segments"].append({
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text
                    })
            
            self.logger.info(f"Transcription completed: {len(result['segments'])} segments")
            return result
            
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise
    
    def _merge_transcription_and_speakers(self, transcript: Dict[str, Any], 
                                        speaker_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge transcription segments with speaker information."""
        merged_result = []
        
        if not transcript.get("segments") or not speaker_segments:
            # If no diarization, just return transcription with unknown speaker
            for segment in transcript.get("segments", []):
                merged_result.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": "unknown",
                    "role": "unknown"
                })
            return merged_result
        
        # Merge based on timestamps
        for segment in transcript["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find overlapping speaker segment
            best_speaker = "unknown"
            best_role = "unknown"
            max_overlap = 0
            
            for speaker_seg in speaker_segments:
                # Calculate overlap
                overlap_start = max(segment_start, speaker_seg["start"])
                overlap_end = min(segment_end, speaker_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = speaker_seg["speaker"]
                    best_role = speaker_seg["role"]
            
            merged_result.append({
                "start": segment_start,
                "end": segment_end,
                "text": segment["text"],
                "speaker": best_speaker,
                "role": best_role
            })
        
        return merged_result
    
    def print_conversation(self, merged_result: List[Dict[str, Any]]):
        """Print the conversation in a readable format."""
        print("\n" + "="*50)
        print("CONVERSATION TRANSCRIPT")
        print("="*50)
        
        for segment in merged_result:
            timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
            role = segment['role'].upper()
            text = segment['text'].strip()
            print(f"{timestamp} {role}: {text}")
        
        print("="*50 + "\n")