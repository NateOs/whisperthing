import json
import time
import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Set

from dotenv import load_dotenv
from src.simple_transcriber import SimpleTranscriber
from src.utils import setup_logging

class FileMonitor:
    """Monitor a folder for new audio files and process them automatically."""
    
    def __init__(self, watch_folder: str, check_interval: int = 60):
        self.watch_folder = Path(watch_folder)
        self.check_interval = check_interval
        self.processed_files: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        self.transcriber = None
        
        # Processed files log
        self.processed_files_log = self.watch_folder / "processed_files.log"
        
        # Supported audio file extensions
        self.audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        
        # Ensure watch folder exists
        self.watch_folder.mkdir(parents=True, exist_ok=True)
        
        # Load previously processed files
        self._load_processed_files()
    
    def initialize_transcriber(self):
        """Initialize the transcriber if not already done."""
        if self.transcriber is None:
            try:
                self.logger.info("Initializing transcriber...")
                self.transcriber = SimpleTranscriber()
                self.logger.info("Transcriber initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize transcriber: {e}")
                raise
    
    def _load_processed_files(self):
        """Load the list of previously processed files from the log."""
        if self.processed_files_log.exists():
            try:
                with open(self.processed_files_log, 'r', encoding='utf-8') as f:
                    for line in f:
                        file_path = line.strip()
                        if file_path:  # Skip empty lines
                            self.processed_files.add(file_path)
                
                self.logger.info(f"Loaded {len(self.processed_files)} previously processed files from log")
                
            except Exception as e:
                self.logger.error(f"Error loading processed files log: {e}")
                self.processed_files = set()  # Start fresh if log is corrupted
        else:
            self.logger.info("No processed files log found, starting fresh")
    
    def _save_processed_file(self, file_path: str):
        """Add a processed file to the log and memory set."""
        try:
            # Add to memory set
            self.processed_files.add(file_path)
            
            # Append to log file
            with open(self.processed_files_log, 'a', encoding='utf-8') as f:
                f.write(f"{file_path}\n")
                
            self.logger.debug(f"Added to processed files log: {Path(file_path).name}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed file to log: {e}")
    
    def _cleanup_processed_files_log(self):
        """Remove entries from the log for files that no longer exist in the watch folder."""
        if not self.processed_files_log.exists():
            return
        
        try:
            # Get all current files in watch folder
            current_files = set(str(f) for f in self.watch_folder.iterdir() if f.is_file())
            
            # Filter out files that still exist
            remaining_files = self.processed_files & current_files
            
            # Write back only the files that still exist
            with open(self.processed_files_log, 'w', encoding='utf-8') as f:
                for file_path in remaining_files:
                    f.write(f"{file_path}\n")
            
            # Update in-memory set
            self.processed_files = remaining_files
            
            self.logger.info(f"Cleaned up processed files log. Remaining entries: {len(remaining_files)}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up processed files log: {e}")
    
    def scan_for_new_files(self) -> list:
        """Scan the watch folder for new audio files."""
        new_files = []
        
        try:
            for file_path in self.watch_folder.iterdir():
                if (file_path.is_file() and 
                    file_path.suffix.lower() in self.audio_extensions and
                    str(file_path) not in self.processed_files):
                    new_files.append(file_path)
            
        except Exception as e:
            self.logger.error(f"Error scanning folder {self.watch_folder}: {e}")
        
        return new_files
    
    def process_file(self, file_path: Path):
        """Process a single audio file."""
        try:
            self.logger.info(f"Processing new file: {file_path.name}")
            
            # Initialize transcriber if not already done
            self.initialize_transcriber()
            
            # Process the audio file
            result = self.transcriber.process_audio(str(file_path))
            
            if result.get("status") == "completed":
                self.logger.info(f"Successfully processed: {file_path.name}")
                
                # Save to output folder as JSON
                try:
                    output_dir = Path(os.getenv("OUTPUT_DIRECTORY", "./output"))
                    output_dir.mkdir(exist_ok=True)
                    
                    output_file = output_dir / f"{file_path.stem}_analysis.json"
                    
                    # Remove internal status info before saving
                    result_to_save = {k: v for k, v in result.items() if k not in ['status', 'chunks_used']}
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(result_to_save, f, indent=2, ensure_ascii=False)
                    
                    self.logger.info(f"Analysis saved to: {output_file}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to save JSON output: {e}")
                
                # Print summary
                keyword_count = len(result.get("keyword_detections", []))
                saved_to_db = result.get("saved_to_database", False)
                
                print(f"✓ {file_path.name} - Keywords found: {keyword_count}, Saved to DB: {saved_to_db}")
                
                # Mark as processed
                self._save_processed_file(str(file_path))
                
            else:
                self.logger.error(f"Failed to process: {file_path.name} - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
    
    def start_monitoring(self):
        """Start the continuous file monitoring loop."""
        self.logger.info(f"Starting file monitor on folder: {self.watch_folder}")
        self.logger.info(f"Check interval: {self.check_interval} seconds")
        self.logger.info(f"Supported extensions: {', '.join(self.audio_extensions)}")
        
        print(f"🔍 Monitoring folder: {self.watch_folder}")
        print(f"⏱️  Check interval: {self.check_interval} seconds")
        print("Press Ctrl+C to stop monitoring\n")
        
        try:
            while True:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Clean up processed files log
                self._cleanup_processed_files_log()
                
                # Scan for new files
                new_files = self.scan_for_new_files()
                
                if new_files:
                    print(f"[{current_time}] Found {len(new_files)} new file(s)")
                    
                    for file_path in new_files:
                        self.process_file(file_path)
                else:
                    print(f"[{current_time}] No new files found")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("File monitoring stopped by user")
            print("\n🛑 Monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            print(f"\n❌ Monitoring error: {e}")

def process_single_file(file_path: str):
    """Process a single audio file."""
    logger = logging.getLogger(__name__)
    
    try:
        transcriber = SimpleTranscriber()
        result = transcriber.process_audio(file_path)
        
        if result.get("status") == "completed":
            logger.info(f"Successfully processed: {file_path}")
            
            # Print conversation if available
            if result.get("merged_result"):
                transcriber.print_conversation(result["merged_result"])
            
            # Print keyword summary
            keyword_detections = result.get("keyword_detections", [])
            if keyword_detections:
                print(f"\n📊 Keywords found: {len(keyword_detections)}")
                for detection in keyword_detections[:10]:  # Show first 10
                    print(f"  • {detection['keyword']} at {detection['timestamp']:.1f}s ({detection['speaker']})")
        else:
            logger.error(f"Failed to process: {file_path} - {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error processing file: {e}")

def process_folder(folder_path: str):
    """Process all audio files in a folder once."""
    logger = logging.getLogger(__name__)
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder does not exist: {folder_path}")
        return
    
    audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    audio_files = [f for f in folder.iterdir() 
                   if f.is_file() and f.suffix.lower() in audio_extensions]
    
    if not audio_files:
        logger.info(f"No audio files found in: {folder_path}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    transcriber = SimpleTranscriber()
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"\nProcessing {i}/{len(audio_files)}: {file_path.name}")
        result = transcriber.process_audio(str(file_path))
        
        if result.get("status") == "completed":
            keyword_count = len(result.get("keyword_detections", []))
            print(f"✓ Completed - Keywords found: {keyword_count}")
        else:
            print(f"❌ Failed - {result.get('error', 'Unknown error')}")

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
    parser = argparse.ArgumentParser(description="Call Transcription Analyzer")
    parser.add_argument("--audio-file", help="Path to audio file to process")
    parser.add_argument("--audio-folder", help="Path to folder containing audio files")
    parser.add_argument("--watch-folder", help="Path to folder to monitor for new files")
    parser.add_argument("--check-interval", type=int, default=60, 
                       help="Interval in seconds to check for new files (default: 60)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # If no arguments provided, default to monitoring current directory
    if not any([args.audio_file, args.audio_folder, args.watch_folder]):
        print("No arguments provided. Starting file monitoring in current directory...")
        args.watch_folder = "./input"
    
    if args.watch_folder:
        # Start continuous monitoring
        monitor = FileMonitor(args.watch_folder, args.check_interval)
        monitor.start_monitoring()
        
    elif args.audio_file:
        # Process single file
        process_single_file(args.audio_file)
        
    elif args.audio_folder:
        # Process folder once
        process_folder(args.audio_folder)
    
    # Original main logic for batch processing (kept for compatibility)
    else:
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