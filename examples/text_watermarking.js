/**
 * AI Watermark - Text Watermarking Example
 * Based on Ucaretron Inc. patent technology
 * 
 * This example demonstrates how to apply visual watermarks to AI-generated text
 * based on confidence scores.
 */

class TextWatermarker {
  constructor(options = {}) {
    this.options = {
      highConfidenceThreshold: options.highConfidenceThreshold || 0.8,
      lowConfidenceThreshold: options.lowConfidenceThreshold || 0.5,
      highConfidenceStyle: options.highConfidenceStyle || { color: '#000000' }, // Black
      mediumConfidenceStyle: options.mediumConfidenceStyle || { color: '#3a86ff' }, // Blue
      lowConfidenceStyle: options.lowConfidenceStyle || { color: '#ff5a5f', italic: true }, // Red + italic
      ...options
    };
  }

  /**
   * Apply watermark styles to text based on confidence scores
   * @param {string} text - The text to watermark
   * @param {Array} confidenceScores - Array of confidence scores (0-1) for each segment
   * @returns {string} HTML with appropriate styling
   */
  applyWatermark(text, confidenceScores) {
    // Split text into segments that can be individually styled
    const segments = this._splitIntoSegments(text);
    
    if (segments.length !== confidenceScores.length) {
      throw new Error('Number of text segments must match number of confidence scores');
    }
    
    let watermarkedText = '';
    
    // Apply appropriate style to each segment based on its confidence score
    for (let i = 0; i < segments.length; i++) {
      const confidence = confidenceScores[i];
      const segment = segments[i];
      
      if (confidence >= this.options.highConfidenceThreshold) {
        watermarkedText += this._applyStyle(segment, this.options.highConfidenceStyle);
      } else if (confidence >= this.options.lowConfidenceThreshold) {
        watermarkedText += this._applyStyle(segment, this.options.mediumConfidenceStyle);
      } else {
        watermarkedText += this._applyStyle(segment, this.options.lowConfidenceStyle);
      }
    }
    
    return watermarkedText;
  }
  
  /**
   * Split text into logical segments (e.g., sentences or paragraphs)
   * @param {string} text - The full text
   * @returns {Array} Array of text segments
   */
  _splitIntoSegments(text) {
    // This is a simplified implementation - in practice, you might use NLP
    // to identify sentence boundaries more accurately
    return text.split(/(?<=[.!?])\s+/);
  }
  
  /**
   * Apply CSS styles to a text segment
   * @param {string} segment - Text segment
   * @param {Object} style - Style object with CSS properties
   * @returns {string} HTML with inline styles
   */
  _applyStyle(segment, style) {
    let styleString = '';
    
    for (const [property, value] of Object.entries(style)) {
      if (property === 'italic' && value === true) {
        // Special case for italic
        return `<em style="${styleString}">${segment}</em>`;
      } else {
        styleString += `${this._camelToKebab(property)}: ${value}; `;
      }
    }
    
    return `<span style="${styleString}">${segment}</span>`;
  }
  
  /**
   * Convert camelCase to kebab-case for CSS properties
   * @param {string} str - camelCase string
   * @returns {string} kebab-case string
   */
  _camelToKebab(str) {
    return str.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase();
  }
}

// Example usage
const watermarker = new TextWatermarker();

const sampleText = `Electrochemical enzymeless sensors have shown great potential in various applications. 
These sensors have been widely used in healthcare monitoring and diagnostics. 
The detection of hazardous gases such as nitrogen dioxide is a key application area. 
Some studies suggest these sensors may have applications in neural interfaces, but more research is needed.
With further development, these sensors can become important tools for environmental monitoring.`;

const confidenceScores = [
  0.92, // High confidence
  0.85, // High confidence
  0.75, // Medium confidence
  0.40, // Low confidence
  0.88  // High confidence
];

const watermarkedHTML = watermarker.applyWatermark(sampleText, confidenceScores);

// In a browser environment, you could insert this into the DOM
// document.getElementById('output').innerHTML = watermarkedHTML;

console.log("AI Watermark applied successfully based on confidence scores.");
console.log("See HTML output for styled text with visual watermarks.");

module.exports = { TextWatermarker };