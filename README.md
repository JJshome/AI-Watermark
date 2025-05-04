# AI-Watermark

Implementation of AI watermarking techniques for audio and video based on patent by Ucaretron Inc. This repository provides tools for embedding imperceptible watermarks in AI-generated content to indicate confidence levels or identify AI-generated portions.

## Background

This project implements concepts from the patent "Expression Method of inference from Artificial Intelligence" by Ucaretron Inc., which proposes methods for marking AI-generated content based on confidence levels or to distinguish between real and AI-generated content.

The core concept is to apply different types of markings to content based on:
- Confidence thresholds of AI inferences
- Identifying AI-generated vs. real content
- Continuous confidence values representation

## Features

This repository includes:

1. **Audio Watermarking**: Embeds imperceptible frequency watermarks into audio files
2. **Video Watermarking**: Applies frame-by-frame watermarking techniques to video content
3. **SVG Visualizations**: Illustrations of how the watermarking techniques work
4. **Extraction Tools**: Methods to detect and visualize embedded watermarks
5. **Sample Media**: Examples with and without watermarks

## How It Works

### Audio Watermarking

<svg width="600" height="300" viewBox="0 0 600 300" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="600" height="300" fill="#f8f9fa" />
  
  <!-- Title -->
  <text x="300" y="30" font-family="Arial" font-size="20" text-anchor="middle" font-weight="bold">Audio Watermarking Process</text>
  
  <!-- Original Audio Waveform -->
  <path d="M50,150 C70,100 90,200 110,150 C130,100 150,200 170,150 C190,100 210,200 230,150" 
        stroke="#2980b9" stroke-width="2" fill="none" />
  <text x="140" y="220" font-family="Arial" font-size="14" text-anchor="middle">Original Audio</text>
  
  <!-- Process Arrow -->
  <path d="M250,150 L290,150" stroke="#333" stroke-width="2" />
  <polygon points="290,145 300,150 290,155" fill="#333" />
  
  <!-- Watermark Signal -->
  <path d="M310,180 C315,175 320,185 325,180 C330,175 335,185 340,180 C345,175 350,185 355,180" 
        stroke="#e74c3c" stroke-width="1" fill="none" stroke-dasharray="3,3" />
  <text x="335" y="200" font-family="Arial" font-size="12" text-anchor="middle" fill="#e74c3c">Inaudible Watermark</text>
  <path d="M335,160 L335,180" stroke="#e74c3c" stroke-width="1" stroke-dasharray="3,3" />
  
  <!-- Process Arrow -->
  <path d="M370,150 L410,150" stroke="#333" stroke-width="2" />
  <polygon points="410,145 420,150 410,155" fill="#333" />
  
  <!-- Watermarked Audio Waveform -->
  <path d="M430,150 C450,95 470,205 490,150 C510,95 530,205 550,150" 
        stroke="#2980b9" stroke-width="2" fill="none" />
  <path d="M430,150 C450,105 470,195 490,150 C510,105 530,195 550,150" 
        stroke="#e74c3c" stroke-width="1" fill="none" stroke-dasharray="2,2" />
  <text x="490" y="220" font-family="Arial" font-size="14" text-anchor="middle">Watermarked Audio</text>
  
  <!-- Legend -->
  <rect x="50" y="250" width="15" height="2" fill="#2980b9" />
  <text x="75" y="254" font-family="Arial" font-size="12" alignment-baseline="middle">Original Signal</text>
  
  <rect x="200" y="250" width="15" height="2" fill="#e74c3c" />
  <text x="225" y="254" font-family="Arial" font-size="12" alignment-baseline="middle">Watermark Signal</text>
  
  <rect x="350" y="250" width="15" height="0" stroke="#e74c3c" stroke-width="1" stroke-dasharray="2,2" />
  <text x="375" y="254" font-family="Arial" font-size="12" alignment-baseline="middle">Combined Signal</text>
  
  <!-- Footer Notes -->
  <text x="300" y="280" font-family="Arial" font-size="10" text-anchor="middle" fill="#555">
    Based on Ucaretron Inc. patent for marking AI-generated content
  </text>
</svg>

Our audio watermarking technique embeds imperceptible frequency markers in specific frequency bands (typically 18-20kHz) that are outside normal human hearing range but detectable with analysis tools. The implementation supports:

- Adding watermarks to mark low-confidence segments
- Including metadata about AI generation within the watermark
- Adjustable strength to balance imperceptibility and robustness
- License information embedding

### Video Watermarking

<svg width="600" height="400" viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <rect width="600" height="400" fill="#f8f9fa" />
  
  <!-- Title -->
  <text x="300" y="30" font-family="Arial" font-size="20" text-anchor="middle" font-weight="bold">Video Watermarking Process</text>
  
  <!-- Original Video Frames -->
  <rect x="50" y="80" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="60" y="90" width="80" height="60" fill="#d6eaf8" />
  <text x="100" y="125" font-family="Arial" font-size="12" text-anchor="middle">Frame 1</text>
  
  <rect x="50" y="170" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="60" y="180" width="80" height="60" fill="#d6eaf8" />
  <text x="100" y="215" font-family="Arial" font-size="12" text-anchor="middle">Frame 2</text>
  
  <rect x="50" y="260" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="60" y="270" width="80" height="60" fill="#d6eaf8" />
  <text x="100" y="305" font-family="Arial" font-size="12" text-anchor="middle">Frame 3</text>
  
  <text x="100" y="360" font-family="Arial" font-size="14" text-anchor="middle">Original Frames</text>
  
  <!-- Process Arrows -->
  <path d="M170,125 L220,125" stroke="#333" stroke-width="2" />
  <polygon points="220,120 230,125 220,130" fill="#333" />
  
  <path d="M170,215 L220,215" stroke="#333" stroke-width="2" />
  <polygon points="220,210 230,215 220,220" fill="#333" />
  
  <path d="M170,305 L220,305" stroke="#333" stroke-width="2" />
  <polygon points="220,300 230,305 220,310" fill="#333" />
  
  <!-- Watermark Process Box -->
  <rect x="230" y="170" width="120" height="100" fill="#fff" stroke="#333" rx="10" ry="10" />
  <text x="290" y="200" font-family="Arial" font-size="14" text-anchor="middle" font-weight="bold">Watermarking</text>
  <text x="290" y="220" font-family="Arial" font-size="12" text-anchor="middle">- Color Histogram</text>
  <text x="290" y="240" font-family="Arial" font-size="12" text-anchor="middle">- Frame Markers</text>
  <text x="290" y="260" font-family="Arial" font-size="12" text-anchor="middle">- Pixel Encoding</text>
  
  <!-- Process Arrows -->
  <path d="M350,125 L400,125" stroke="#333" stroke-width="2" />
  <polygon points="400,120 410,125 400,130" fill="#333" />
  
  <path d="M350,215 L400,215" stroke="#333" stroke-width="2" />
  <polygon points="400,210 410,215 400,220" fill="#333" />
  
  <path d="M350,305 L400,305" stroke="#333" stroke-width="2" />
  <polygon points="400,300 410,305 400,310" fill="#333" />
  
  <!-- Watermarked Video Frames -->
  <rect x="410" y="80" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="420" y="90" width="80" height="60" fill="#d6eaf8" />
  <path d="M420,90 L500,150" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <path d="M500,90 L420,150" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <text x="460" y="125" font-family="Arial" font-size="12" text-anchor="middle">Frame 1</text>
  <rect x="480" y="100" width="10" height="10" fill="#e74c3c" fill-opacity="0.2" />
  
  <rect x="410" y="170" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="420" y="180" width="80" height="60" fill="#d6eaf8" />
  <path d="M420,180 L500,240" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <path d="M500,180 L420,240" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <text x="460" y="215" font-family="Arial" font-size="12" text-anchor="middle">Frame 2</text>
  <rect x="430" y="190" width="10" height="10" fill="#e74c3c" fill-opacity="0.2" />
  
  <rect x="410" y="260" width="100" height="80" fill="#fff" stroke="#333" />
  <rect x="420" y="270" width="80" height="60" fill="#d6eaf8" />
  <path d="M420,270 L500,330" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <path d="M500,270 L420,330" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <text x="460" y="305" font-family="Arial" font-size="12" text-anchor="middle">Frame 3</text>
  <rect x="470" y="310" width="10" height="10" fill="#e74c3c" fill-opacity="0.2" />
  
  <text x="460" y="360" font-family="Arial" font-size="14" text-anchor="middle">Watermarked Frames</text>
  
  <!-- Legend -->
  <rect x="100" y="380" width="15" height="10" fill="#d6eaf8" />
  <text x="125" y="388" font-family="Arial" font-size="12" alignment-baseline="middle">Original Content</text>
  
  <rect x="250" y="380" width="15" height="10" fill="#e74c3c" fill-opacity="0.2" />
  <text x="275" y="388" font-family="Arial" font-size="12" alignment-baseline="middle">Watermark</text>
  
  <path d="M390,380 L420,389" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <path d="M420,380 L390,389" stroke="#e74c3c" stroke-width="0.5" stroke-opacity="0.3" />
  <text x="460" y="388" font-family="Arial" font-size="12" alignment-baseline="middle">Encoded Pattern</text>
</svg>

Our video watermarking applies several techniques:

1. **Frame-based watermarking**: Each frame receives an imperceptible marker
2. **Color histogram modification**: Subtle changes to color distributions that persist through compression
3. **Inter-frame watermarking**: Information encoded across frame sequences
4. **Metadata embedding**: Confidence levels and AI attribution information
5. **Extraction tools**: For detecting and analyzing the embedded watermarks

## Usage

### Audio Watermarking

```python
# Add watermark to audio file
python audio_watermark.py --input input.wav --output watermarked.wav --strength 0.05

# Extract and analyze watermark
python audio_extract.py --input watermarked.wav --visualize
```

### Video Watermarking

```python
# Add watermark to video file
python video_watermark.py --input input.mp4 --output watermarked.mp4 --method histogram

# Extract and analyze watermark
python video_extract.py --input watermarked.mp4 --output analysis.png
```

## Installation

```bash
# Clone the repository
git clone https://github.com/JJshome/AI-Watermark.git
cd AI-Watermark

# Install dependencies
pip install -r requirements.txt
```

## License

This project is for educational and research purposes only. Implementation is based on the patent by Ucaretron Inc. for methods of marking AI-generated content.

## Acknowledgements

- Based on the patent "Expression Method of inference from Artificial Intelligence" by Ucaretron Inc.
- Uses open-source libraries for audio and video processing
- Inspired by techniques for digital watermarking and steganography
