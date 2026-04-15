# Audio/Video Transcription Service
Flask-based transcription service using OpenAI's Whisper model with automatic speaker detection.

## Features

- 🎙️ **Multi-format support**: MP3, MP4, WAV, M4A, FLAC, AVI, MOV, WebM
- 👥 **Speaker diarization**: Automatic detection of 2-6 speakers
- 🚀 **Flexible processing**: CPU or GPU (NVIDIA) acceleration
- 🎯 **Multiple models**: Base (fast), Medium (balanced), Large-v3 (accurate)
- 🌐 **Web interface**: Drag-and-drop file upload with progress tracking
- ⚡ **Real-time progress**: Live updates with cancellation support

## Quick Start

### Option 1: CPU Mode (Works on All Systems)

**Recommended for MacBooks and Windows PCs without NVIDIA GPU**

```bash
# Build and start the service
docker-compose up --build

# Access the web interface
open http://localhost:8080
```

### Option 2: GPU Mode (NVIDIA GPUs only)

**Requirements:**
- NVIDIA GPU
- [NVIDIA Docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

```bash
# Build and start with GPU support
docker-compose -f docker-compose.gpu.yml up --build

# Access the web interface
open http://localhost:8080
```

## Usage

1. Open http://localhost:8080 in your browser
2. Select processing device (CPU/GPU) and model size (Base/Medium/Large)
3. Upload or drag-and-drop an audio/video file
4. Click "Transcribe" and wait for processing
5. View, copy, or download the transcript

## Output Format

Transcripts are saved in `./outputs/` with timestamps and speaker labels:

```
[00:00:00 - 00:00:04] Speaker 1: Hello, how are you?
[00:00:04 - 00:00:07] Speaker 2: I'm doing well, thanks!
```

## Supported File Formats

- **Audio**: MP3, WAV, M4A, FLAC
- **Video**: MP4, AVI, MOV, WebM

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload file for transcription
- `GET /download/<filename>` - Download transcript
- `POST /cancel/<task_id>` - Cancel running transcription
- `GET /health` - System health check

## Configuration

### Model Selection
- **Base**: Fastest, good for simple conversations
- **Medium**: Balanced speed and accuracy (default)
- **Large-v3**: Highest accuracy, slower

### Device Selection
- **CPU**: Works everywhere, slower (recommended for MacBooks)
- **GPU**: 5-10x faster, requires NVIDIA GPU

## Stopping the Service

```bash
# Stop and remove containers
docker-compose down

# For GPU version
docker-compose -f docker-compose.gpu.yml down
```

## Troubleshooting

### "could not select device driver nvidia"
- You're trying to use GPU mode without NVIDIA GPU
- **Solution**: Use the default `docker-compose.yml` (CPU mode)

### Slow transcription on Mac
- This is normal - Macs use CPU processing
- **Tips**:
  - Use "Base" model for faster results
  - Consider using smaller file chunks
  - Expect ~1-2 minutes per minute of audio on CPU

### Port already in use
- Change the port in `docker-compose.yml`:
  ```yaml
  ports:
    - "8081:5000"  # Use 8081 instead of 8080
  ```

## License

This project uses OpenAI's Whisper model. Check their licensing terms for commercial use.
