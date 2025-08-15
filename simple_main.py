import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.transcription import SimpleTranscriptionService
from src.audio_utils import validate_audio_quality

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check required environment variables
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        logger.error("Please set your OPENAI_API_KEY in the .env file")
        return
    
    # Initialize transcription service
    service = SimpleTranscriptionService()
    
    # Process audio files
    input_dir = Path(os.getenv("INPUT_DIRECTORY", "./input"))
    output_dir = Path(os.getenv("OUTPUT_DIRECTORY", "./output"))
    
    # Create directories if they don't exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Find audio files
    audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))
    
    if not audio_files:
        logger.info(f"No audio files found in {input_dir}")
        return
    
    # Process each file
    for audio_file in audio_files:
        try:
            # Validate audio quality first
            is_valid, message = validate_audio_quality(audio_file)
            if not is_valid:
                logger.warning(f"Skipping {audio_file}: {message}")
                continue
                
            logger.info(f"Processing: {audio_file}")
            result = service.transcribe_audio(audio_file)
            
            # Save result to JSON
            output_file = output_dir / f"{audio_file.stem}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Result saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")

if __name__ == "__main__":
    main()