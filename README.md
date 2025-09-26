# 🎙️ Call Transcription Analyzer

A powerful Python application designed to transcribe and analyze Spanish call recordings, offering detailed insights through speaker identification, keyword detection, and comprehensive analytics. Built with OpenAI's Whisper and advanced audio processing capabilities.

## ✨ Features

- 🎯 **Smart Transcription**: High-accuracy Spanish transcription using OpenAI Whisper
- 👥 **Speaker Recognition**: Automatically distinguishes between agents and customers
- 🔍 **Keyword Analytics**: Detects and timestamps important phrases and keywords
- 📊 **Metrics & Insights**: Generates detailed call analytics and statistics
- 💾 **Data Persistence**: Stores results in MS SQL Server for analysis
- ⏰ **Automation Ready**: Built-in scheduling support via Windows Task Scheduler
- 🔄 **Continuous Processing**: Monitors folders for new recordings

## 📋 Prerequisites

- 🐍 Python 3.8 or higher
- 🖥️ Windows OS (for scheduled execution)
- 🗄️ MS SQL Server (optional, for data storage)
- 🎵 Audio files in WAV format
- 🔑 OpenAI API key
- 🎫 Hugging Face token (for speaker recognition)

## 🚀 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/nateos/whisperthing.git
   cd whisperthing
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   # For NVIDIA GPU support (recommended)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # Install other dependencies
   pip install -r requirements.txt
   ```

4. Set up environment configuration:

   ```bash
   # Copy example configuration
   cp .env.example .env

   # Edit .env with your credentials
   # Required: OPENAI_API_KEY, HUGGINGFACE_TOKEN
   # Optional: Database settings
   ```

5. Configure database (optional):

   ```bash
   # Ensure SQL Server is running
   # Run the setup script in SQL Server Management Studio
   database_setup.sql

   # Test connection
   python test_db_connection.py
   ```

## 💻 Usage

### Command Line Interface

Process recordings in different ways:

```bash
# Process a single file
python main.py --audio-file "path/to/recording.wav"

# Process all files in a directory
python main.py --audio-folder "path/to/recordings"

# Monitor a folder for new files
python main.py --watch-folder "path/to/watch" --check-interval 60

# Simple mode: just run and process files in input/
python main.py
```

### 🔄 Automated Processing

The included `run_analyzer.bat` script provides automated execution:

```bash
# Run manually
run_analyzer.bat

# Or schedule with Windows Task Scheduler
```

### ⏰ Setting Up Scheduled Execution

1. Open Windows Task Scheduler
2. Create a new Basic Task
3. Configure the schedule (e.g., daily at specific time)
4. Action: "Start a program"
5. Program: Select `run_analyzer.bat`
6. Start in: Set to project root (e.g., `C:\github\whisperthing`)
7. Complete and save

## 📊 Data Storage

### Database Schema

The application uses four main tables:

| Table                | Description                                       |
| -------------------- | ------------------------------------------------- |
| `calls`              | Main call records and metadata                    |
| `transcriptions`     | Detailed transcription segments with speaker info |
| `keyword_detections` | Detected keywords with timestamps                 |
| `call_metrics`       | Analytics and statistics per call                 |

### 📈 Analysis Output

For each processed recording, you get:

- 📝 Full transcription with speaker identification
- 🎯 Keyword occurrences with exact timestamps
- 📊 Call metrics (talk time, word counts, politeness)
- 🔄 Processing metadata and status

### 🔍 Keyword Detection

Default Spanish keywords include:

**Politeness:**

- "gracias", "muchas gracias"
- "por favor", "disculpe"
- "buenos días", "buenas tardes"

**Service:**

- "ayuda", "servicio"
- "problema", "consulta"
- "factura", "pago"

## 🛠️ Troubleshooting

### Common Issues

1. **First Run Delay**

   - Initial run downloads Whisper model
   - May take several minutes depending on connection

2. **Speaker Recognition**

   - Requires valid Hugging Face token
   - Check `.env` file configuration

3. **Database Connectivity**

   - Run `test_db_connection.py`
   - Verify SQL Server credentials
   - Ensure SQL Server is running

4. **Audio Processing**
   - Only WAV format supported
   - Files over 25MB are split automatically
   - Check available disk space

## ⚙️ Configuration

### Environment Variables

Create `.env` file with these settings:

```ini
# API Keys (Required)
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here

# Database Settings (Optional)
DB_DRIVER={ODBC Driver 17 for SQL Server}
DB_SERVER=localhost
DB_DATABASE=CallAnalysis
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_TRUSTED_CONNECTION=no

# Processing Options
WHISPER_MODEL_SIZE=base
WHISPER_LANGUAGE=es
SPEAKER_SEPARATION_ENABLED=true
CONFIDENCE_THRESHOLD=0.7

# Custom Keywords (Optional)
ANALYSIS_KEYWORDS=gracias,por favor,disculpe,buenos días

# Directories
INPUT_DIRECTORY=./input
OUTPUT_DIRECTORY=./output

# Performance (Optional)
CUDA_VISIBLE_DEVICES=0  # For GPU support
```

## 📁 Project Structure

```
whisperthing/
├── input/               # Place audio files here
├── output/             # Results and analysis
├── src/                # Source code
│   ├── analysis.py     # Call analysis
│   ├── audio_utils.py  # Audio processing
│   ├── database.py     # Data storage
│   └── models.py       # Data models
├── .env                # Configuration
└── main.py            # Entry point
```

## 📜 License

This project is privately licensed. All rights reserved.

## 🙏 Acknowledgments

- OpenAI for Whisper
- Pyannote.audio team
- PyTorch community
- Python SQL Server team
