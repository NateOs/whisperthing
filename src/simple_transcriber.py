import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import warnings
from dotenv import load_dotenv
from openai import OpenAI
import json
import math
import sys
import shutil
from enum import Enum

from src.models import TranscriptionResult, TranscriptionSegment
from .analysis import CallAnalyzer
from .config import DatabaseConfig
from .database import DatabaseManager

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

class TranscriptionStatus(Enum):
    """Enumeration for transcription process status."""
    PENDING = "pending"
    SPLITTING = "splitting"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"


class SimpleTranscriber:
    """Enhanced transcription service with file splitting and better speaker detection."""
    
    def __init__(self):
        load_dotenv()
        self.logger = logging.getLogger(__name__)
        
        # File size limits (OpenAI Whisper API limit is 25MB)
        self.max_size_mb = 25.0
        self.max_size_bytes = self.max_size_mb * 1024 * 1024
        
        # Process tracking
        self.chunks_directories: Set[Path] = set()
        self.file_statuses: Dict[str, TranscriptionStatus] = {}
        self.processing_errors: Dict[str, str] = {}
        
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
        
        # Initialize database manager
        self.db_manager = None
        self._setup_database()
        
        self.logger.info("Simple transcriber initialized successfully")
        if AUDIO_PROCESSING_AVAILABLE:
            self.logger.info("Audio processing (pydub) is available")
        else:
            self.logger.warning("Audio processing (pydub) is not available - large files cannot be split")
        
        print("Loading CallAnalyzer...")   
        self.analyzer = CallAnalyzer()
    
    def _setup_database(self):
        """Set up database connection."""
        try:
            db_config = DatabaseConfig.from_env()
            self.db_manager = DatabaseManager(db_config)
            
            # Test connection
            if self.db_manager.test_connection():
                self.logger.info("Database connection established successfully")
            else:
                self.logger.warning("Database connection test failed - results will not be saved to database")
                self.db_manager = None
        except Exception as e:
            self.logger.warning(f"Failed to initialize database: {e} - results will not be saved to database")
            self.db_manager = None

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
            
            # Move pipeline to GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    self.pipeline.to(device)
                    self.logger.info("Speaker diarization pipeline loaded successfully on GPU")
                else:
                    self.logger.info("Speaker diarization pipeline loaded successfully on CPU (GPU not available)")
            except ImportError:
                self.logger.info("Speaker diarization pipeline loaded successfully on CPU (PyTorch not available)")
            
        except Exception as e:
            try:
                # Fallback to older model
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=hf_token
                )
                # Also try to move fallback to GPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        self.pipeline.to(torch.device("cuda"))
                        self.logger.info("Fallback speaker diarization pipeline loaded on GPU")
                    else:
                        self.logger.info("Fallback speaker diarization pipeline loaded on CPU")
                except ImportError:
                    self.logger.info("Fallback speaker diarization pipeline loaded on CPU")
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
        
        # Track this chunks directory
        self.chunks_directories.add(output_dir)
        
        try:
            # Load audio file
            audio = AudioSegment.from_wav(str(input_path))
            
            # Calculate file size and determine if splitting is needed
            file_size = input_path.stat().st_size
            if file_size <= self.max_size_bytes:
                # File doesn't need splitting, clean up empty chunks dir
                if output_dir.exists() and not any(output_dir.iterdir()):
                    output_dir.rmdir()
                    self.chunks_directories.discard(output_dir)
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
            # Remove from tracking if splitting failed
            self.chunks_directories.discard(output_dir)
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
    
    def _run_diarization(self, audio_file_path: str, num_speakers: int = None, min_speakers: int = None, max_speakers: int = None) -> List[Dict[str, Any]]:
        """Run speaker diarization on audio file with speaker constraints."""
        if not self.pipeline:
            self.logger.warning("Diarization pipeline not available")
            return []
        
        try:
            # Prepare pipeline parameters
            pipeline_params = {}
            if num_speakers is not None:
                pipeline_params["num_speakers"] = num_speakers
            elif min_speakers is not None or max_speakers is not None:
                if min_speakers is not None:
                    pipeline_params["min_speakers"] = min_speakers
                if max_speakers is not None:
                    pipeline_params["max_speakers"] = max_speakers
            
            # For call center scenarios, typically 2 speakers (agent + customer)
            if not pipeline_params:
                pipeline_params["num_speakers"] = 2
            
            # Pre-load audio and run diarization
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sample_rate != 16000:
                    import torchaudio.transforms as T
                    resampler = T.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                
                diarization = self.pipeline({"waveform": waveform, "sample_rate": 16000}, **pipeline_params)
            except ImportError:
                diarization = self.pipeline(audio_file_path, **pipeline_params)
            
            # Process results (same as before)
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
                
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += segment["duration"]
            
            # Assign roles
            if len(speaker_times) >= 2:
                sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
                speaker_mapping = {
                    sorted_speakers[0][0]: "agent",
                    sorted_speakers[1][0]: "customer"
                }
                for i, (speaker, _) in enumerate(sorted_speakers[2:], start=2):
                    speaker_mapping[speaker] = f"speaker_{i+1}"
            else:
                speaker_mapping = {list(speaker_times.keys())[0]: "agent"} if speaker_times else {}
            
            for segment in speaker_segments:
                segment["role"] = speaker_mapping.get(segment["speaker"], "unknown")
            
            self.logger.info(f"Diarization completed: {len(speaker_segments)} segments found")
            return speaker_segments
            
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return []
    
    def _run_diarization_with_progress(self, audio_file_path: str) -> List[Dict[str, Any]]:
        """Run speaker diarization with progress monitoring."""
        if not self.pipeline:
            self.logger.warning("Diarization pipeline not available")
            return []
        
        try:
            # Import progress hook
            try:
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                
                # Pre-load audio in memory for faster processing
                try:
                    import torchaudio
                    waveform, sample_rate = torchaudio.load(audio_file_path)
                    # Ensure mono audio at 16kHz
                    if waveform.shape[0] > 1:
                        waveform = waveform.mean(dim=0, keepdim=True)
                    if sample_rate != 16000:
                        import torchaudio.transforms as T
                        resampler = T.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    
                    # Run with progress monitoring
                    with ProgressHook() as hook:
                        diarization = self.pipeline({"waveform": waveform, "sample_rate": 16000}, hook=hook)
                        
                except ImportError:
                    # Fallback without pre-loading
                    with ProgressHook() as hook:
                        diarization = self.pipeline(audio_file_path, hook=hook)
                        
            except ImportError:
                # Fallback without progress monitoring
                return self._run_diarization(audio_file_path)
            
            # Process results (same as before)
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
                
                if speaker not in speaker_times:
                    speaker_times[speaker] = 0
                speaker_times[speaker] += segment["duration"]
            
            # Assign roles based on speaking time
            if len(speaker_times) >= 2:
                sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
                speaker_mapping = {
                    sorted_speakers[0][0]: "agent",
                    sorted_speakers[1][0]: "customer"
                }
                for i, (speaker, _) in enumerate(sorted_speakers[2:], start=2):
                    speaker_mapping[speaker] = f"speaker_{i+1}"
            else:
                speaker_mapping = {list(speaker_times.keys())[0]: "agent"} if speaker_times else {}
            
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
    
    def process_audio(self, audio_file_path: str, num_speakers: int = 2, save_to_db: bool = True) -> Dict[str, Any]:
        """Process audio file with diarization and transcription."""
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        file_path_str = str(audio_path)
        self._update_file_status(file_path_str, TranscriptionStatus.PENDING)
        
        try:
            self.logger.info(f"Processing audio file: {audio_path}")
            
            # Step 1: Check file size and split if necessary
            self._update_file_status(file_path_str, TranscriptionStatus.SPLITTING)
            if not self._check_file_size(audio_path):
                self.logger.info(f"File size exceeds limit ({self.max_size_mb}MB), attempting to split {audio_path}")
                try:
                    chunks = self._split_audio_file(audio_path)
                except ImportError:
                    self.logger.warning("Cannot split large file, attempting to process anyway...")
                    chunks = [audio_path]
            else:
                chunks = [audio_path]
            
            # Step 2: Run transcription on each chunk
            self._update_file_status(file_path_str, TranscriptionStatus.TRANSCRIBING)
            transcriptions = []
            for i, chunk in enumerate(chunks, 1):
                self.logger.info(f"Transcribing chunk {i}/{len(chunks)}: {chunk}")
                transcript = self._run_transcription(str(chunk))
                transcriptions.append(transcript)
            
            # Step 3: Merge transcriptions
            self._update_file_status(file_path_str, TranscriptionStatus.MERGING)
            merged_transcript = self._merge_transcriptions(transcriptions, chunks)
            
            # Step 4: Run optimized diarization on original file
            self._update_file_status(file_path_str, TranscriptionStatus.DIARIZING)
            speaker_segments = self._run_diarization(str(audio_path), num_speakers=num_speakers)
            
            # Step 5: Merge based on timestamps
            merged_result = self._merge_transcription_and_speakers(merged_transcript, speaker_segments)
            
            # Step 6: Analyze keywords
            self._update_file_status(file_path_str, TranscriptionStatus.ANALYZING)
            keyword_detections = []
            analysis_result = None
            
            try:
                # convert merged_result to a format suitable for analysis
                transcription_segments = []
                for segment in merged_result:
                    transcription_segments.append(TranscriptionSegment(
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                        speaker=segment["role"],  # Use role (agent/customer) instead of speaker ID
                        confidence=getattr(segment, 'confidence', 0.8)
                    ))
                    
                # Create TranscriptionResult object
                audio_file_size = audio_path.stat().st_size
                total_duration = merged_result[-1]["end"] if merged_result else 0.0
                
                transcription_result = TranscriptionResult(
                    text=merged_transcript.get("text", ""), 
                    segments=transcription_segments,
                    language=merged_transcript.get("language", "es"),
                    duration=total_duration,
                    file_size=audio_file_size
                )
                
                # Run keyword analysis
                analysis_result = self.analyzer.analyze_call(transcription_result)
                keyword_detections = analysis_result.keyword_detections
                
                self.logger.info(f"Keyword analysis completed: {len(keyword_detections)} keywords detected")
                
            except Exception as e:
                self.logger.warning(f"Keyword analysis failed: {e}")

            # Step 7: Save to database if enabled and available
            call_id = None
            if save_to_db and self.db_manager and analysis_result:
                try:
                    call_id = self.db_manager.save_call_analysis(
                        audio_file_path=str(audio_path),
                        transcription=transcription_result,
                        analysis=analysis_result
                    )
                    self.logger.info(f"Results saved to database with call ID: {call_id}")
                except Exception as e:
                    self.logger.error(f"Failed to save results to database: {e}")

            self._update_file_status(file_path_str, TranscriptionStatus.COMPLETED)

            result = {
                "file_path": str(audio_path),
                "call_id": call_id,  # Include database ID if saved
                "speaker_segments": speaker_segments,
                "transcript": merged_transcript,
                "merged_result": merged_result,
                "keyword_detections": [
                    {
                        "keyword": kd.keyword,
                        "timestamp": kd.timestamp,
                        "speaker": kd.speaker,
                        "context": kd.context,
                        "confidence": kd.confidence
                    } for kd in keyword_detections
                ],
                "analysis_summary": {
                    "total_keywords_found": len(keyword_detections),
                    "keywords_by_speaker": self._group_keywords_by_speaker(keyword_detections),
                    "analysis_time": analysis_result.analysis_time if analysis_result else 0
                } if keyword_detections else None,
                "status": TranscriptionStatus.COMPLETED.value,
                "chunks_used": len(chunks) > 1,
                "saved_to_database": call_id is not None
            }
        except Exception as e:
            self.logger.error(f"Error processing audio file {audio_file_path}: {e}")
            self._update_file_status(file_path_str, TranscriptionStatus.FAILED, error=str(e))
            result = {
                "file_path": str(audio_path),
                "error": str(e),
                "status": TranscriptionStatus.FAILED.value,
                "saved_to_database": False
            }
        return result
    
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
    
    def _cleanup_chunks(self, chunks_dir: Path):
        """Clean up the chunks directory after processing."""
        if chunks_dir and chunks_dir.exists() and chunks_dir.name == "chunks":
            try:
                shutil.rmtree(chunks_dir)
                self.logger.info(f"Cleaned up chunks directory: {chunks_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up chunks directory {chunks_dir}: {e}")
    
    def _update_file_status(self, file_path: str, status: TranscriptionStatus, error: str = None):
        """Update the status of a file being processed."""
        self.file_statuses[file_path] = status
        if error:
            self.processing_errors[file_path] = error
        self.logger.info(f"File {Path(file_path).name}: {status.value}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get a summary of the processing status."""
        status_counts = {}
        for status in TranscriptionStatus:
            status_counts[status.value] = sum(1 for s in self.file_statuses.values() if s == status)
        
        return {
            "total_files": len(self.file_statuses),
            "status_counts": status_counts,
            "chunks_directories": [str(d) for d in self.chunks_directories],
            "errors": self.processing_errors.copy(),
            "all_completed": all(status == TranscriptionStatus.COMPLETED for status in self.file_statuses.values()),
            "any_failed": any(status == TranscriptionStatus.FAILED for status in self.file_statuses.values())
        }
    def _group_keywords_by_speaker(self, keyword_detections: List) -> Dict[str, int]:
        """Group keyword detections by speaker."""
        speaker_counts = {}
        for detection in keyword_detections:
            speaker = detection.speaker
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        return speaker_counts
    
    def cleanup_all_chunks(self, force: bool = False) -> Dict[str, Any]:
        """
        Clean up all chunks directories.
        
        Args:
            force: If True, clean up even if not all files completed successfully
        
        Returns:
            Dictionary with cleanup results
        """
        summary = self.get_processing_summary()
        cleanup_results = {
            "attempted": [],
            "successful": [],
            "failed": [],
            "skipped_reason": None
        }
        
        # Check if we should proceed with cleanup
        if not force and not summary["all_completed"]:
            failed_files = [f for f, s in self.file_statuses.items() if s == TranscriptionStatus.FAILED]
            pending_files = [f for f, s in self.file_statuses.items() if s != TranscriptionStatus.COMPLETED and s != TranscriptionStatus.FAILED]
            
            cleanup_results["skipped_reason"] = {
                "message": "Not all files completed successfully",
                "failed_files": failed_files,
                "pending_files": pending_files
            }
            self.logger.warning("Skipping cleanup: Not all files completed successfully. Use force=True to cleanup anyway.")
            return cleanup_results
        
        # Proceed with cleanup
        for chunks_dir in self.chunks_directories.copy():
            cleanup_results["attempted"].append(str(chunks_dir))
            try:
                if chunks_dir.exists():
                    shutil.rmtree(chunks_dir)
                    cleanup_results["successful"].append(str(chunks_dir))
                    self.logger.info(f"Cleaned up chunks directory: {chunks_dir}")
                else:
                    cleanup_results["successful"].append(str(chunks_dir))
                    self.logger.info(f"Chunks directory already removed: {chunks_dir}")
                
                # Remove from tracking
                self.chunks_directories.discard(chunks_dir)
                
            except Exception as e:
                cleanup_results["failed"].append({"directory": str(chunks_dir), "error": str(e)})
                self.logger.warning(f"Failed to clean up chunks directory {chunks_dir}: {e}")
        
        self.logger.info(f"Cleanup completed. Successful: {len(cleanup_results['successful'])}, Failed: {len(cleanup_results['failed'])}")
        return cleanup_results