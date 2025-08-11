import os
import logging
from pathlib import Path
from typing import List, Tuple
from pydub import AudioSegment
import math

class AudioProcessor:
    """Utility class for audio file processing."""
    
    def __init__(self, max_size_mb: float = 24.0):
        self.max_size_mb = max_size_mb
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.logger = logging.getLogger(__name__)
    
    def check_file_size(self, file_path: Path) -> bool:
        """Check if file is within size limits."""
        file_size = file_path.stat().st_size
        return file_size <= self.max_size_bytes
    
    def split_audio_file(self, input_path: Path, output_dir: Path = None) -> List[Path]:
        """Split large audio file into smaller chunks."""
        if output_dir is None:
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
    
    def merge_transcriptions(self, transcriptions: List[dict], chunk_paths: List[Path]) -> dict:
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
