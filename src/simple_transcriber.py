import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import json
import math
import sys

# Suppress deprecation warnings from audio libraries
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pydub")
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*ffmpeg.*")
warnings.filterwarnings("ignore", message=".*std.*degrees of freedom.*")

# Handle audioop import for different Python versions
try:
    if sys.version_info >= (3, 13):
        import audioop_lts as audioop
        # Make audioop_lts available as audioop for pydub
        sys.modules['audioop'] = audioop
    else:
        import audioop
    AUDIOOP_AVAILABLE = True
except ImportError:
    AUDIOOP_AVAILABLE = False
    audioop = None

try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
except ImportError as e:
    DIARIZATION_AVAILABLE = False
    Pipeline = None
    logging.warning(f"Diarization unavailable: {e}")

try:
    # Now try to import pydub after setting up audioop
    from pydub import AudioSegment
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError as e:
    AUDIO_PROCESSING_AVAILABLE = False
    AudioSegment = None
    logging.warning(f"Audio processing unavailable: {e}")

class SimpleTranscriber:
    """Enhanced transcription service with file splitting and better speaker detection."""
    
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        
        # File size limits (OpenAI Whisper API limit is 25MB)
        self.max_size_mb = 25.0
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            raise ValueError("Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=api_key)
        
        # Initialize diarization pipeline
        self.pipeline = None
        if DIARIZATION_AVAILABLE:
            self._setup_diarization()
        
        # Load keywords for analysis
        self.keywords = self._load_keywords()
        
        self.logger.info("Simple transcriber initialized successfully")
        if AUDIO_PROCESSING_AVAILABLE:
            self.logger.info("Audio processing (pydub) is available")
        else:
            self.logger.warning("Audio processing (pydub) is not available - large files cannot be split")
    
    def _setup_diarization(self):
        """Set up speaker diarization pipeline."""
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token or hf_token == "your_huggingface_token_here":
            self.logger.warning("No HuggingFace token provided, diarization will be disabled")
            return
        
        try:
            # Try the newer model first
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            self.logger.info("Speaker diarization pipeline loaded successfully")
        except Exception as e:
            try:
                # Fallback to older model
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
                self.logger.info("Fallback speaker diarization pipeline loaded")
            except Exception as e2:
                self.logger.warning(f"Failed to load diarization pipeline: {e}")
                self.logger.info("Please visit https://hf.co/pyannote/speaker-diarization-3.1 to accept user conditions")
    
    def _load_keywords(self) -> List[str]:
        """Load keywords for analysis."""
        custom_keywords = os.getenv("ANALYSIS_KEYWORDS", "")
        if custom_keywords:
            return [k.strip() for k in custom_keywords.split(",")]
        
        # Default Spanish keywords for call analysis
        return [
            "gracias", "muchas gracias", "por favor", "disculpe", "perdón",
            "problema", "ayuda", "servicio", "cliente", "factura", "pago",
            "cancelar", "activar", "desactivar", "consulta", "queja",
            "buenos días", "buenas tardes", "buenas noches", "de nada",
            "con permiso", "si me permite", "entiendo", "perfecto"
        ]
    
    def _check_file_size(self, file_path: Path) -> bool:
        """Check if file is within size limits."""
        file_size = file_path.stat().st_size
        return file_size <= self.max_size_bytes
    
    def _split_audio_file(self, input_path: Path) -> List[Path]:
        """Split large audio file into smaller chunks."""
        if not AUDIO_PROCESSING_AVAILABLE:
            self.logger.error("Cannot split audio file: pydub is not available")
            self.logger.info("File is too large for OpenAI API. Please manually split the file or install pydub dependencies.")
            raise ImportError("pydub is required for audio splitting but is not available")
        
        output_dir = input_path.parent / "chunks"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Load audio file
            audio = AudioSegment.from_wav(str(input_path))
            
            # Calculate file size and determine if splitting is needed
            file_size = input_path.stat().st_size
            if file_size <= self.max_size_bytes:
                return [input_path]
            
            # Calculate chunk duration based on file size
            total_duration_ms = len(audio)
            num_chunks = math.ceil(file_size / self.max_size_bytes)
            chunk_duration_ms = total_duration_ms // num_chunks
            
            # Add 10% overlap between chunks to avoid cutting words
            overlap_ms = int(chunk_duration_ms * 0.1)
            
            chunks = []
            base_name = input_path.stem
            
            for i in range(num_chunks):
                start_ms = max(0, i * chunk_duration_ms - (overlap_ms if i > 0 else 0))
                end_ms = min(total_duration_ms, (i + 1) * chunk_duration_ms + overlap_ms)
                
                chunk = audio[start_ms:end_ms]
                chunk_path = output_dir / f"{base_name}_chunk_{i+1:02d}.wav"
                
                chunk.export(str(chunk_path), format="wav")
                chunks.append(chunk_path)
                
                self.logger.info(f"Created chunk {i+1}/{num_chunks}: {chunk_path}")
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to split audio file {input_path}: {e}")
            raise
    
    def _run_transcription(self, audio_file_path: str) -> Dict[str, Any]:
        """Run OpenAI Whisper transcription."""
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="es",
                    response_format="verbose_json",
                    timestamp_granularities=["segment"]
                )
            
            return {
                "text": transcript.text,
                "language": getattr(transcript, 'language', 'es'),
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text
                    }
                    for seg in getattr(transcript, 'segments', [])
                ]
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI transcription failed: {e}")
            raise
    
    def _run_diarization(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """Run speaker diarization on audio file."""
        if not self.pipeline:
            self.logger.warning("Diarization pipeline not available")
            return []
        
        try:
            diarization = self.pipeline(audio_file_path)
            
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
                
                # Add additional speakers as "other" if present
                for i, (speaker, _) in enumerate(sorted_speakers[2:], start=2):
                    speaker_mapping[speaker] = f"speaker_{i+1}"
            else:
                # Single speaker
                speaker_mapping = {list(speaker_times.keys())[0]: "agent"} if speaker_times else {}
            
            # Add role to segments
            for segment in speaker_segments:
                segment["role"] = speaker_mapping.get(segment["speaker"], "unknown")
            
            self.logger.info(f"Diarization completed: {len(speaker_segments)} segments found")
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return []
    
    def _assign_speakers_simple(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple speaker assignment when diarization is not available."""
        # Alternate between agent and customer based on segment length and pauses
        assigned_segments = []
        current_speaker = "agent"  # Start with agent
        
        for i, segment in enumerate(segments):
            # Simple heuristic: longer segments are more likely to be agent
            # Short responses are more likely to be customer
            segment_duration = segment["end"] - segment["start"]
            text_length = len(segment["text"].split())
            
            # If segment is very short (< 2 seconds) and few words, likely customer
            if segment_duration < 2.0 and text_length < 5:
                current_speaker = "customer"
            # If segment is long (> 10 seconds) or many words, likely agent
            elif segment_duration > 10.0 or text_length > 20:
                current_speaker = "agent"
            # Otherwise, alternate
            else:
                current_speaker = "customer" if current_speaker == "agent" else "agent"
            
            segment_with_speaker = segment.copy()
            segment_with_speaker["speaker"] = current_speaker
            assigned_segments.append(segment_with_speaker)
        
        return assigned_segments
    
    def process_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Process audio file with diarization and transcription."""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        self.logger.info(f"Processing audio file: {audio_path}")
        
        # Check file size and split if necessary
        if not self._check_file_size(audio_path):
            self.logger.info(f"File size exceeds limit ({self.max_size_mb}MB), attempting to split {audio_path}")
            try:
                chunks = self._split_audio_file(audio_path)
            except ImportError:
                # If we can't split, try to process anyway (will likely fail at OpenAI API)
                self.logger.warning("Cannot split large file, attempting to process anyway...")
                chunks = [audio_path]
        else:
            chunks = [audio_path]
        
        # Step 1: Run transcription on each chunk
        transcriptions = []
        for chunk in chunks:
            self.logger.info(f"Transcribing chunk: {chunk}")
            transcript = self._run_transcription(str(chunk))
            transcriptions.append(transcript)
        
        # Step 2: Merge transcriptions
        merged_transcript = self._merge_transcriptions(transcriptions, chunks)
        
        # Step 3: Run diarization on original file
        speaker_segments = self._run_diarization(str(audio_path))
        
        # Step 4: Merge based on timestamps
        merged_result = self._merge_transcription_and_speakers(merged_transcript, speaker_segments)
        
        return {
            "file_path": str(audio_path),
            "speaker_segments": speaker_segments,
            "transcript": merged_transcript,
            "merged_result": merged_result
        }
    
    def _merge_transcriptions(self, transcriptions: List[dict], chunk_paths: List[Path]) -> dict:
        """Merge transcriptions from multiple chunks."""
        merged_segments = []
        total_offset = 0.0
        
        for i, (transcription, chunk_path) in enumerate(zip(transcriptions, chunk_paths)):
            segments = transcription.get("segments", [])
            
            for segment in segments:
                # Adjust timestamps based on chunk position
                adjusted_segment = segment.copy()
                adjusted_segment["start"] += total_offset
                adjusted_segment["end"] += total_offset
                adjusted_segment["chunk_id"] = i + 1
                adjusted_segment["chunk_file"] = str(chunk_path)
                
                merged_segments.append(adjusted_segment)
            
            # Calculate offset for next chunk (subtract overlap)
            if i < len(transcriptions) - 1:
                chunk_duration = segments[-1]["end"] if segments else 0
                total_offset += chunk_duration * 0.9  # 90% to account for overlap
        
        # Merge all text
        full_text = " ".join([t.get("text", "") for t in transcriptions])
        
        return {
            "text": full_text,
            "segments": merged_segments,
            "chunks_processed": len(transcriptions),
            "language": transcriptions[0].get("language", "es") if transcriptions else "es"
        }
    
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