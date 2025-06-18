import React, { useState } from 'react';
import axios from 'axios';
import TranslationForm from './components/TranslationForm';
import ImageViewer from './components/ImageViewer';
import LoadingSpinner from './components/LoadingSpinner';
import ErrorMessage from './components/ErrorMessage';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface TranslationResponse {
  translated_images: string[];
  original_count: number;
  success: boolean;
  message: string;
}

function App() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [translatedImages, setTranslatedImages] = useState<string[]>([]);
  const [progress, setProgress] = useState<string>('');
  const [currentImage, setCurrentImage] = useState<number>(0);
  const [totalImages, setTotalImages] = useState<number>(0);

  const handleTranslate = async (webtoonUrl: string) => {
    setUrl(webtoonUrl);
    setLoading(true);
    setError(null);
    setTranslatedImages([]);
    setProgress('Initializing translation...');
    setCurrentImage(0);
    setTotalImages(0);

          try {
        // Start progress polling
        const progressInterval = setInterval(async () => {
          try {
            const statusResponse = await axios.get(`${API_BASE_URL}/health`);
            // In a real implementation, you'd have a status endpoint
            // For now, we'll just show connection is alive
          } catch (e) {
            // Ignore errors during progress check
          }
        }, 2000);

        const response = await axios.post<TranslationResponse>(
          `${API_BASE_URL}/api/translate-manhwa`,
          { url: webtoonUrl },
          {
            timeout: 900000, // 15 minute timeout for OCR processing
            onDownloadProgress: (progressEvent) => {
              if (progressEvent.loaded && progressEvent.total) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                setProgress(`Processing images (${percentCompleted}% downloaded)...`);
              }
            }
          }
        );

        clearInterval(progressInterval);

              if (response.data.success) {
          setTranslatedImages(response.data.translated_images);
          setTotalImages(response.data.original_count);
          setProgress('');
          setCurrentImage(response.data.original_count);
        } else {
          throw new Error(response.data.message || 'Translation failed');
        }
    } catch (err) {
      console.error('Translation error:', err);
      if (axios.isAxiosError(err)) {
        if (err.response) {
          setError(err.response.data.detail || 'Translation failed. Please try again.');
        } else if (err.request) {
          setError('Cannot connect to the server. Please make sure the backend is running.');
        } else {
          setError('An unexpected error occurred. Please try again.');
        }
      } else {
        setError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setLoading(false);
      setProgress('');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <h1 className="text-3xl font-bold text-gray-900">
            Manhwa Translator
          </h1>
          <p className="text-gray-600 mt-1">
            Translate Korean webtoons to English while preserving the visual style
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Translation Form */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <TranslationForm 
            onSubmit={handleTranslate} 
            loading={loading}
            initialUrl={url}
          />
          
          {/* Progress Indicator */}
          {loading && (
            <div className="mt-4 space-y-2">
              <div className="text-center text-sm text-gray-600">
                {progress || 'Starting translation process...'}
              </div>
              {totalImages > 0 && (
                <div className="text-center text-xs text-gray-500">
                  Processing up to 30 images from this chapter
                </div>
              )}
              {currentImage > 0 && totalImages > 0 && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(currentImage / totalImages) * 100}%` }}
                  ></div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && <ErrorMessage message={error} onDismiss={() => setError(null)} />}

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-12">
            <LoadingSpinner />
            <p className="mt-4 text-gray-600 text-center">
              Translating your manhwa... This may take a few minutes.
            </p>
            <p className="mt-2 text-sm text-gray-500 text-center">
              Each image goes through OCR detection, AI translation, text removal, and text rendering.
            </p>
            {totalImages > 0 && (
              <p className="mt-2 text-sm font-medium text-blue-600">
                Processing {Math.min(10, totalImages)} images (limited for testing)
              </p>
            )}
          </div>
        )}

        {/* Results */}
        {!loading && translatedImages.length > 0 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-2">
              Translated Manhwa ({translatedImages.length} panels)
            </h2>
            <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-md">
              <p className="text-sm text-green-800">
                ✅ Successfully processed {translatedImages.length} panels from Return of the Mount Hua Sect with context-aware AI translation!
                {totalImages > 30 && (
                  <span className="block mt-1 text-green-700">
                    Note: Limited to first 30 images for processing. Full chapter had {totalImages} images.
                  </span>
                )}
              </p>
            </div>
            <ImageViewer images={translatedImages} />
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <p className="text-center text-gray-400">
            © 2024 Manhwa Translator. For educational purposes only.
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App; 