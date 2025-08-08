import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.simple_transcriber import SimpleTranscriber

def setup_logging():
    """Setup basic logging."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize transcriber
        transcriber = SimpleTranscriber()
        
        # Get directories from environment
        input_dir = Path(os.getenv("INPUT_DIRECTORY", "./input"))
        output_dir = Path(os.getenv("OUTPUT_DIRECTORY", "./output"))
        
        # Create directories if they don't exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Find audio files
        audio_extensions = ["*.wav", "*.mp3", "*.m4a", "*.flac"]
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(input_dir.glob(ext))
        
        if not audio_files:
            logger.info(f"No audio files found in {input_dir}")
            logger.info(f"Supported formats: {', '.join(audio_extensions)}")
            return
        
        logger.info(f"Found {len(audio_files)} audio file(s) to process")
        
        # Process each file
        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {audio_file.name}")
                
                # Process the audio
                result = transcriber.process_audio(str(audio_file))
                
                # Print conversation to console
                transcriber.print_conversation(result["merged_result"])
                
                # Save detailed result to JSON
                output_file = output_dir / f"{audio_file.stem}_analysis.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Detailed analysis saved to: {output_file}")
                
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()