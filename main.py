import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.simple_transcriber import SimpleTranscriber

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('transcription.log')
        ]
    )

def print_processing_summary(transcriber: SimpleTranscriber, logger):
    """Print a summary of the processing status."""
    summary = transcriber.get_processing_summary()
    
    logger.info("=" * 50)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total files: {summary['total_files']}")
    
    for status, count in summary['status_counts'].items():
        if count > 0:
            logger.info(f"  {status.upper()}: {count}")
    
    if summary['errors']:
        logger.info("\nERRORS:")
        for file_path, error in summary['errors'].items():
            logger.error(f"  {Path(file_path).name}: {error}")
    
    if summary['chunks_directories']:
        logger.info(f"\nChunks directories created: {len(summary['chunks_directories'])}")
        for chunks_dir in summary['chunks_directories']:
            logger.info(f"  {chunks_dir}")
    
    logger.info("=" * 50)
    
    return summary

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    transcriber = None
    
    try:
        # Initialize transcriber
        transcriber = SimpleTranscriber()
        
        # Get directories from environment
        input_dir = Path(os.getenv("INPUT_DIRECTORY", "./input"))
        output_dir = Path(os.getenv("OUTPUT_DIRECTORY", "./output"))
        
        # Get database saving preference from environment
        save_to_db = os.getenv("SAVE_TO_DATABASE", "true").lower() == "true"
        logger.info(f"Database saving: {'enabled' if save_to_db else 'disabled'}")
        
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
        successful_files = 0
        failed_files = 0
        saved_to_db_count = 0
        
        for audio_file in audio_files:
            try:
                logger.info(f"Starting processing: {audio_file.name}")
                
                # Process the audio with database saving option
                result = transcriber.process_audio(str(audio_file), save_to_db=save_to_db)
                
                # Track database saves
                if result.get("saved_to_database", False):
                    saved_to_db_count += 1
                    logger.info(f"  Saved to database with call ID: {result.get('call_id')}")
                
                # Print conversation to console if available
                if hasattr(transcriber, 'print_conversation'):
                    transcriber.print_conversation(result["merged_result"])
                
                # Save detailed result to JSON
                output_file = output_dir / f"{audio_file.stem}_analysis.json"
                
                # Remove internal status info before saving
                result_to_save = {k: v for k, v in result.items() if k not in ['status', 'chunks_used']}
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_to_save, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Successfully processed: {audio_file.name}")
                logger.info(f"  Analysis saved to: {output_file}")
                successful_files += 1
                
            except Exception as e:
                logger.error(f"✗ Failed to process {audio_file.name}: {e}")
                failed_files += 1
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        logger.info("BATCH PROCESSING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Successfully processed: {successful_files} files")
        logger.info(f"Failed: {failed_files} files")
        if save_to_db:
            logger.info(f"Saved to database: {saved_to_db_count} files")
        
        # Get detailed processing summary
        summary = print_processing_summary(transcriber, logger)
        
        # Cleanup chunks based on processing results
        if summary['all_completed']:
            logger.info("\n🧹 All files processed successfully. Cleaning up chunks...")
            cleanup_result = transcriber.cleanup_all_chunks()
            
            if cleanup_result['successful']:
                logger.info(f"✓ Successfully cleaned up {len(cleanup_result['successful'])} chunks directories")
            
            if cleanup_result['failed']:
                logger.warning(f"⚠ Failed to clean up {len(cleanup_result['failed'])} chunks directories")
                for failed_cleanup in cleanup_result['failed']:
                    logger.warning(f"  {failed_cleanup['directory']}: {failed_cleanup['error']}")
        
        elif summary['any_failed']:
            logger.warning("\n⚠ Some files failed to process. Chunks will be preserved for debugging.")
            logger.info("You can manually clean up chunks later by calling transcriber.cleanup_all_chunks(force=True)")
            
            # Optionally, ask user if they want to force cleanup anyway
            force_cleanup = os.getenv("FORCE_CLEANUP_ON_FAILURE", "false").lower() == "true"
            if force_cleanup:
                logger.info("🧹 FORCE_CLEANUP_ON_FAILURE is enabled. Cleaning up chunks anyway...")
                cleanup_result = transcriber.cleanup_all_chunks(force=True)
                logger.info(f"Forced cleanup completed: {len(cleanup_result['successful'])} successful, {len(cleanup_result['failed'])} failed")
        
        else:
            logger.info("\n📋 Processing completed with mixed results. Check the summary above.")
        
        logger.info("\n🎉 All processing operations completed!")
        
    except Exception as e:
        logger.error(f"💥 Application error: {e}")
        
        # Emergency cleanup if something went wrong
        if transcriber and hasattr(transcriber, 'chunks_directories'):
            emergency_cleanup = os.getenv("EMERGENCY_CLEANUP", "true").lower() == "true"
            if emergency_cleanup and transcriber.chunks_directories:
                logger.info("🚨 Performing emergency cleanup of chunks directories...")
                try:
                    cleanup_result = transcriber.cleanup_all_chunks(force=True)
                    logger.info(f"Emergency cleanup: {len(cleanup_result['successful'])} directories cleaned")
                except Exception as cleanup_error:
                    logger.error(f"Emergency cleanup failed: {cleanup_error}")

if __name__ == "__main__":
    main()