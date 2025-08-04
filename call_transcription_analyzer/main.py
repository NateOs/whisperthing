import argparse
import logging
import sys
from pathlib import Path
from src.config import Config
from src.database import DatabaseManager
from src.transcription import TranscriptionService
from src.analysis import CallAnalyzer
from src.utils import setup_logging

def main():
    """Main entry point for the call transcription analyzer."""
    parser = argparse.ArgumentParser(description='Call Transcription and Analysis Tool')
    parser.add_argument('--audio-file', type=str, help='Path to the audio file to process')
    parser.add_argument('--audio-folder', type=str, help='Path to folder containing audio files')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config(args.config)
        
        # Initialize services
        db_manager = DatabaseManager(config.database)
        transcription_service = TranscriptionService(config.whisper)
        analyzer = CallAnalyzer(config.analysis)
        
        # Process audio files
        audio_files = []
        if args.audio_file:
            audio_files = [Path(args.audio_file)]
        elif args.audio_folder:
            audio_folder = Path(args.audio_folder)
            audio_files = list(audio_folder.glob("*.wav"))
        else:
            # Default: process all files in configured input directory
            input_dir = Path(config.processing.input_directory)
            audio_files = list(input_dir.glob("*.wav"))
        
        if not audio_files:
            logger.warning("No audio files found to process")
            return
        
        logger.info(f"Processing {len(audio_files)} audio files")
        
        for audio_file in audio_files:
            try:
                logger.info(f"Processing: {audio_file}")
                
                # Transcribe audio
                transcription_result = transcription_service.transcribe_call(audio_file)
                
                # Analyze transcription
                analysis_result = analyzer.analyze_call(transcription_result)
                
                # Save to database
                call_id = db_manager.save_call_analysis(
                    audio_file_path=str(audio_file),
                    transcription=transcription_result,
                    analysis=analysis_result
                )
                
                logger.info(f"Successfully processed call ID: {call_id}")
                
                # Move processed file to completed directory if configured
                if config.processing.move_completed:
                    completed_dir = Path(config.processing.completed_directory)
                    completed_dir.mkdir(exist_ok=True)
                    audio_file.rename(completed_dir / audio_file.name)
                
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")
                continue
        
        logger.info("Processing completed")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()