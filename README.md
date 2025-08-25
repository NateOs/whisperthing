# Call Transcription Analyzer

A Python application that transcribes Spanish call recordings using OpenAI Whisper, performs speaker diarization, and analyzes calls for specific keywords with timestamps.

## Features

- **Audio Transcription**: Uses OpenAI Whisper for accurate Spanish transcription
- **Speaker Separation**: Identifies agent vs customer using pyannote.audio
- **Keyword Detection**: Finds specific keywords with precise timestamps
- **Database Storage**: Saves results to MS SQL Server database
- **Windows Executable**: Can be compiled to .exe for Windows Server deployment
- **Task Scheduler Ready**: Designed to run via Windows Task Scheduler

## Requirements

- Python 3.8+
- Windows Server (for .exe deployment)
- MS SQL Server
- Audio files in .wav format

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd call_transcription_analyzer
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up database:
   - Run `database_setup.sql` in your SQL Server
   - Update `config.json` with your database credentials

5. Configure Hugging Face token (for speaker diarization):
   ```bash
   set HUGGINGFACE_TOKEN=your_token_here
   ```

## Configuration

Edit `config.json` to configure:
- Database connection settings
- Whisper model settings
- Keywords to detect
- Input/output directories

## Usage

### Command Line

Process a single file:
```bash
python main.py --audio-file "path/to/recording.wav"
```

Process all files in a folder:
```bash
python main.py --audio-folder "path/to/recordings/"
```

Use custom config:
```bash
python main.py --config "custom_config.json"
```

### Building Executable

Run the build script:
```bash
build_exe.bat
```

The executable will be created in the `dist` folder.

### Windows Task Scheduler

1. Create a new task in Task Scheduler
2. Set the executable path to `dist/CallAnalyzer.exe`
3. Add arguments as needed (e.g., `--audio-folder "C:\recordings"`)
4. Configure schedule as required

## Database Schema

The application creates four main tables:
- `calls`: Main call records
- `transcriptions`: Transcribed segments with speaker info
- `keyword_detections`: Detected keywords with timestamps
- `call_metrics`: Calculated metrics per call

## Output

For each processed call, the system stores:
- Complete transcription with speaker separation
- Keyword detections with exact timestamps
- Call metrics (talk time, word counts, politeness score)
- Processing metadata

## Example Keywords

The default configuration includes Spanish politeness keywords:
- "gracias", "muchas gracias"
- "por favor", "disculpe"
- "buenos días", "buenas tardes"
- And more...

## Troubleshooting

1. **Whisper model download**: First run will download the model
2. **Speaker diarization**: Requires Hugging Face token
3. **Database connection**: Ensure SQL Server is accessible
4. **Audio format**: Only .wav files are supported

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
# File: /call_transcription_analyzer/.env.example
# Copy this file to .env and fill in your actual values

# Hugging Face token for speaker diarization
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Database credentials (if not using config.json)
DB_SERVER=localhost
DB_DATABASE=CallAnalysis
DB_USERNAME=your_username
DB_PASSWORD=your_password

# Optional: GPU settings
CUDA_VISIBLE_DEVICES=0

<!-- for nvidia graphics -->
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

<!-- TODO track transcribed files so you dont transcribe them again
save transcriptions to db
run as script on windows or linux
let whisper transcribe to audio language
clear chunks after transcription job completes
 -->