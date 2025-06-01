# AI Watermark API Reference

This document provides a comprehensive reference for the AI Watermark API. The API allows developers to integrate AI content confidence visualization into their applications.

## Overview

The AI Watermark API provides tools for marking AI-generated content with visual indicators based on confidence levels. This helps users distinguish between high-confidence AI inferences and potentially unreliable content.

## Authentication

To use the AI Watermark API, you'll need an API key. Contact the repository owner for licensing and API access information.

```javascript
const AIWatermark = require('ai-watermark');
const watermarker = new AIWatermark({
  apiKey: 'YOUR_API_KEY'
});
```

## Core Modules

### Text Watermarking

#### `watermarkText(text, confidenceScores[, options])`

Applies visual watermarks to text based on confidence scores.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | String | The text to watermark |
| `confidenceScores` | Array | Array of confidence scores (0-1) for each segment |
| `options` | Object | Optional configuration (see below) |

**Options:**

```javascript
{
  highConfidenceThreshold: 0.8,    // Threshold for high confidence (default: 0.8)
  lowConfidenceThreshold: 0.5,     // Threshold for low confidence (default: 0.5)
  highConfidenceStyle: {           // Style for high confidence segments
    color: '#000000'               // Black
  },
  mediumConfidenceStyle: {         // Style for medium confidence segments
    color: '#3a86ff'               // Blue
  },
  lowConfidenceStyle: {            // Style for low confidence segments
    color: '#ff5a5f',
    italic: true
  }
}
```

**Returns:**

HTML string with appropriate styling applied to each text segment.

**Example:**

```javascript
const text = "Electrochemical sensors are used in healthcare. Some applications include neural interfaces.";
const confidenceScores = [0.9, 0.4];

const watermarkedText = watermarker.watermarkText(text, confidenceScores);
```

### Image Watermarking

#### `watermarkImage(image, confidenceMap[, options])`

Applies visual watermarks to images based on pixel-level confidence values.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | Buffer/Array | The image data to watermark |
| `confidenceMap` | Array | 2D array of confidence values (0-1) for each pixel |
| `options` | Object | Optional configuration (see below) |

**Options:**

```javascript
{
  highConfidenceColor: [0, 0, 255],   // Blue for high confidence
  mediumConfidenceColor: [0, 0, 0],   // Black for medium confidence
  lowConfidenceColor: [255, 0, 0],    // Red for low confidence
  highConfidenceThreshold: 0.8,
  lowConfidenceThreshold: 0.5,
  watermarkIntensity: 0.3             // Transparency level
}
```

**Returns:**

Buffer containing the watermarked image data.

**Example:**

```javascript
const fs = require('fs');
const image = fs.readFileSync('input.jpg');
const confidenceMap = generateConfidenceMap(image); // Your function to generate confidence values

const watermarkedImage = watermarker.watermarkImage(image, confidenceMap);
fs.writeFileSync('watermarked.jpg', watermarkedImage);
```

### Audio Watermarking

#### `watermarkAudio(audio, confidenceTimeline[, options])`

Applies audible watermarks to audio based on confidence over time.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `audio` | Buffer | The audio data to watermark |
| `confidenceTimeline` | Array | Array of confidence values (0-1) over time |
| `options` | Object | Optional configuration |

**Returns:**

Buffer containing the watermarked audio data.

### Video Watermarking

#### `watermarkVideo(video, confidenceFrames[, options])`

Applies visual watermarks to video frames based on confidence values.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `video` | Buffer | The video data to watermark |
| `confidenceFrames` | Array | Array of frame-by-frame confidence maps |
| `options` | Object | Optional configuration |

**Returns:**

Buffer containing the watermarked video data.

## Digital Watermarking

### `embedDigitalWatermark(content, metadata[, options])`

Embeds invisible digital watermarks containing confidence metadata.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | Buffer | The content to watermark (image, audio, etc.) |
| `metadata` | Object | Metadata including confidence information |
| `options` | Object | Optional configuration |

**Returns:**

Buffer containing the content with embedded digital watermark.

### `extractDigitalWatermark(content)`

Extracts digital watermark from content.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `content` | Buffer | Content with embedded watermark |

**Returns:**

Object containing extracted metadata.

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages:

```javascript
try {
  const result = await watermarker.watermarkText(text, confidenceScores);
} catch (error) {
  console.error(`Error: ${error.message}`);
  // Handle error appropriately
}
```

## Rate Limits

The API has the following rate limits:

- 100 requests per minute for free tier
- 1000 requests per minute for standard tier
- Custom limits for enterprise tier

## Patent Information

The AI Watermark technology is based on patents by Ucaretron Inc. For licensing information, please contact the repository owner.

---

## Python Utilities & Example Scripts

While the core API above is presented in a JavaScript-like syntax for web-centric applications, this project also includes Python-based utilities and examples for backend processing, testing, and demonstration of watermarking concepts.

### Sample Audio Generation

-   **Function**: `create_sample_audio()` located in `examples/example_usage.py`.
-   **Purpose**: Generates sample WAV audio files with various configurable properties.
-   **Features**:
    -   Waveforms: 'sine', 'square', 'sawtooth', 'noise'.
    -   Configurable frequency (for tonal waveforms), duration, and sample rate.
-   **Usage**: Useful for creating test carriers for audio watermarking experiments or demonstrations. See `examples/example_usage.py` for how it's used in a workflow.

### Audio Comparison Tool

-   **Script**: `compare_audio.py` (located in the project root).
-   **Purpose**: A command-line tool to analyze and compare two audio files (e.g., an original and its watermarked version).
-   **Features**:
    -   Accepts paths to two audio files as command-line arguments.
    -   Performs frequency analysis (e.g., energy at typical watermarking carrier frequencies) on both files and prints a JSON summary.
    -   Provides an interactive CLI menu to play either the original or the watermarked audio file.
    -   OS-aware playback using default system players (`aplay` for Linux, `afplay` for macOS, PowerShell for Windows).
-   **Usage**: `python compare_audio.py <original_audio_path> <watermarked_audio_path>`
-   **Example Workflow**: The `examples/example_usage.py` script demonstrates generating a sample audio, watermarking it, and then provides an example of how `compare_audio.py` could be invoked (though in the example, `compare_audio.py` is called via `subprocess` for an integrated demonstration).