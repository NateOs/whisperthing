from setuptools import setup, find_packages

setup(
    name="call-transcription-analyzer",
    version="1.0.0",
    description="Spanish call transcription and analysis tool using Whisper",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "openai-whisper>=20231117",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "pyodbc>=4.0.39",
        "pydub>=0.25.1",
        "pyannote.audio>=3.1.0",
        "pyannote.core>=5.0.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "call-analyzer=main:main",
        ],
    },
)