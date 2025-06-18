import React from 'react';

const LoadingSpinner: React.FC = () => {
  return (
    <div className="relative">
      <div className="w-16 h-16 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin"></div>
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="w-10 h-10 border-4 border-gray-100 border-b-blue-400 rounded-full animate-spin animate-reverse"></div>
      </div>
    </div>
  );
};

export default LoadingSpinner; 