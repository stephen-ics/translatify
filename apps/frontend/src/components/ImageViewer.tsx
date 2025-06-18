import React, { useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface ImageViewerProps {
  images: string[];
}

const ImageViewer: React.FC<ImageViewerProps> = ({ images }) => {
  const [loadedImages, setLoadedImages] = useState<Set<number>>(new Set());
  const [errorImages, setErrorImages] = useState<Set<number>>(new Set());
  const [downloading, setDownloading] = useState(false);

  const handleImageLoad = (index: number) => {
    setLoadedImages(prev => new Set(prev).add(index));
  };

  const handleImageError = (index: number) => {
    setErrorImages(prev => new Set(prev).add(index));
  };

  const handleDownload = async () => {
    try {
      setDownloading(true);
      console.log(`🎬 Starting download of ${images.length} images...`);
      
      if (images.length === 0) {
        alert('No images to download');
        return;
      }
      
      console.log('📤 Sending images to backend for combination...');
      
      // Call backend to create combined strip
      const response = await axios.post(
        `${API_BASE_URL}/api/download-strip`,
        { images },
        { 
          responseType: 'blob',
          timeout: 60000, // 1 minute timeout
        }
      );
      
      console.log(`📥 Received ${response.data.size} bytes from backend`);
      
      // Create download link
      const blob = new Blob([response.data], { type: 'image/jpeg' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'manhwa_translated_strip.jpg';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
      
      console.log('✅ Download completed successfully!');
      
    } catch (error) {
      console.error('❌ Download failed:', error);
      if (axios.isAxiosError(error)) {
        if (error.response) {
          alert(`Download failed: ${error.response.data.detail || error.response.statusText}`);
        } else if (error.request) {
          alert('Download failed: Could not connect to server');
        } else {
          alert('Download failed: Request setup error');
        }
      } else {
        alert('Download failed: Unknown error');
      }
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-0">
      {/* Vertical scroll container */}
      <div className="max-w-4xl mx-auto">
        {images.map((image, index) => (
          <div key={index} className="relative">
            {/* Loading placeholder */}
            {!loadedImages.has(index) && !errorImages.has(index) && (
              <div className="absolute inset-0 bg-gray-100 animate-pulse flex items-center justify-center">
                <div className="text-gray-400">
                  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <p className="mt-2 text-sm">Loading panel {index + 1}...</p>
                </div>
              </div>
            )}
            
            {/* Error state */}
            {errorImages.has(index) && (
              <div className="bg-red-50 border border-red-200 rounded-md p-8 text-center">
                <svg className="w-12 h-12 text-red-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                        d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p className="text-red-600">Failed to load panel {index + 1}</p>
              </div>
            )}
            
            {/* Image */}
            {!errorImages.has(index) && (
              <img
                src={image}
                alt={`Translated manhwa panel ${index + 1}`}
                className={`w-full h-auto ${loadedImages.has(index) ? 'block' : 'block'}`}
                onLoad={() => handleImageLoad(index)}
                onError={() => handleImageError(index)}
                loading="lazy"
              />
            )}
          </div>
        ))}
      </div>
      
            {/* Download button */}
      <div className="mt-8 text-center space-y-3">
        <button
          onClick={handleDownload}
          disabled={downloading}
          className={`inline-flex items-center px-6 py-3 rounded-md transition-colors font-medium ${
            downloading
              ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
              : 'bg-green-600 text-white hover:bg-green-700 shadow-md hover:shadow-lg'
          }`}
        >
          {downloading ? (
            <>
              <svg className="w-5 h-5 mr-2 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
                Combining {images.length} panels into strip...
            </>
          ) : (
            <>
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Download as Single Strip ({images.length} panels)
            </>
          )}
        </button>
        
        <div className="text-sm text-gray-600 max-w-md mx-auto">
          <p>📱 Downloads all {images.length} translated panels as one long vertical image - perfect for mobile reading!</p>
          <p className="mt-1 text-xs text-gray-500">File will be saved as "manhwa_translated_strip.jpg"</p>
        </div>
      </div>
    </div>
  );
};

export default ImageViewer; 