import React, { useState, FormEvent } from 'react';

interface TranslationFormProps {
  onSubmit: (url: string) => void;
  loading: boolean;
  initialUrl?: string;
}

const TranslationForm: React.FC<TranslationFormProps> = ({ onSubmit, loading, initialUrl = '' }) => {
  const [url, setUrl] = useState(initialUrl);
  const [error, setError] = useState('');

  const validateUrl = (url: string): boolean => {
    try {
      const urlObj = new URL(url);
      return urlObj.hostname.includes('comic.naver.com');
    } catch {
      return false;
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    setError('');

    if (!url.trim()) {
      setError('Please enter a URL');
      return;
    }

    if (!validateUrl(url)) {
      setError('Please enter a valid Naver Webtoon URL');
      return;
    }

    onSubmit(url);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="url" className="block text-sm font-medium text-gray-700 mb-2">
          Naver Webtoon URL
        </label>
        <div className="flex gap-2">
          <input
            type="url"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://comic.naver.com/webtoon/detail?titleId=769209&no=1"
            className="flex-1 px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-colors"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading}
            className={`px-6 py-2 rounded-md font-medium transition-colors ${
              loading
                ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {loading ? 'Converting...' : 'Convert'}
          </button>
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-600">{error}</p>
        )}
      </div>
      
      <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
        <h3 className="text-sm font-semibold text-blue-900 mb-2">How to use:</h3>
        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
          <li>Go to a Naver Webtoon chapter page</li>
          <li>Copy the URL from your browser</li>
          <li>Paste it in the field above</li>
          <li>Click "Convert" and wait for the translation</li>
        </ol>
      </div>
    </form>
  );
};

export default TranslationForm; 