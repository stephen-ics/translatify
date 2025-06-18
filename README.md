# Manhwa Translator - Korean to English

A monorepo application that translates Korean manhwa webtoons into English while preserving the visual style of the text.

## Overview

This application scrapes manhwa images from Naver Webtoon URLs and uses advanced OCR and AI to:
1. **Detect Korean text** in manhwa panels using EasyOCR
2. **Translate text** to natural English using OpenAI GPT-4
3. **Remove original text** using computer vision inpainting
4. **Add English text** with proper styling and positioning
5. **Preserve visual quality** while making manhwa accessible in English

## Architecture

- **Monorepo Structure**: Uses pnpm workspaces
- **Backend**: Python FastAPI server with OCR, AI translation, and image processing
- **Frontend**: React TypeScript application with Vite and Tailwind CSS
- **OCR**: EasyOCR for Korean text detection and extraction
- **AI Translation**: OpenAI GPT-4 for natural Korean-to-English translation
- **Image Processing**: OpenCV for text removal and PIL for text rendering

## Prerequisites

- Node.js >= 18.0.0
- Python >= 3.9
- pnpm >= 8.0.0
- OpenAI API key with GPT-4 Vision access

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd manhwa-site-mvp

# Install pnpm if not already installed
npm install -g pnpm@8.12.0

# Install all dependencies
pnpm install
```

### 2. Backend Setup

```bash
# Navigate to backend
cd apps/backend

# Create Python virtual environment (recommended)
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create .env file from example
cp .env-example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Frontend Setup

The frontend dependencies are already installed via pnpm. No additional setup needed.

### 4. Running the Application

You can run both frontend and backend simultaneously or separately:

#### Option A: Run Both Together (from root directory)

```bash
# From the root directory
pnpm dev
```

#### Option B: Run Separately

**Terminal 1 - Backend:**
```bash
cd apps/backend
# Make sure virtual environment is activated
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd apps/frontend
pnpm dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### API Endpoints:
- `POST /api/translate-manhwa` - Translate a manhwa from Naver Webtoon URL
- `POST /api/download-strip` - Combine translated images into downloadable strip
- `GET /health` - Health check endpoint

## Usage

1. Open the frontend at http://localhost:5173
2. Enter a Naver Webtoon chapter URL (e.g., `https://comic.naver.com/webtoon/detail?titleId=769209&no=1`)
3. Click "Convert" and wait for the translation process (this will now actually translate the text!)
4. The system will process up to 10 images (limited for testing) and for each image:
   - 🔍 Detect Korean text using OCR
   - 🌐 Translate text to English using AI
   - 🎨 Remove original Korean text using inpainting
   - ✍️ Add English text with proper styling and positioning
5. View detailed progress logs in the browser console and backend terminal
6. View the translated manhwa panels below in vertical reading format  
7. **Click "Download as Single Strip"** to download all translated panels as one long vertical image
   - Perfect for mobile reading! 📱
   - Combines all 10 panels into a single JPEG file
   - Watch the backend logs for detailed combining progress
   - File automatically downloads as "manhwa_translated_strip.jpg"

## Project Structure

```
manhwa-site-mvp/
├── apps/
│   ├── backend/          # FastAPI Python backend
│   │   ├── main.py      # Main API server
│   │   ├── requirements.txt
│   │   └── .env-example
│   └── frontend/        # React TypeScript frontend
│       ├── src/
│       │   ├── components/
│       │   ├── App.tsx
│       │   └── main.tsx
│       └── package.json
├── package.json         # Root package.json
├── pnpm-workspace.yaml  # pnpm workspace configuration
└── README.md
```

## AI Prompt Template

The backend uses a sophisticated prompt template to instruct the AI model on how to translate and style-match the text. The key aspects include:

1. **Text Identification**: Identifies all Korean text including speech bubbles, sound effects, and narrative boxes
2. **Translation Quality**: Preserves meaning, tone, and character voice
3. **Visual Matching**: Maintains original font style, size, color, effects, and integration with artwork
4. **Image Generation**: Outputs a new image with Korean text replaced by styled English

## Important Notes

### Current Implementation

✅ **Real Text Translation**: Now implements actual Korean text detection, translation, and replacement
✅ **OCR Detection**: Uses EasyOCR to detect Korean text with high accuracy
✅ **Text Removal**: Uses OpenCV inpainting to remove original text cleanly
✅ **Text Styling**: Adds English text with outlines and proper positioning

### Current Limitations

- **Font Matching**: Uses system fonts rather than matching original manhwa fonts exactly
- **Complex Layouts**: May struggle with very stylized text or complex speech bubble shapes  
- **Processing Time**: OCR and AI translation takes time (several seconds per panel)
- **CORS**: The backend is configured to accept requests only from local frontend URLs
- **Rate Limiting**: Be mindful of OpenAI API rate limits when processing multiple images

### Security Considerations

- Never commit your actual API keys to version control
- The scraping functionality should be used responsibly and in accordance with website terms of service
- This is designed for educational purposes

## Development

### Backend Development

```bash
cd apps/backend
# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
# Run with auto-reload
python -m uvicorn main:app --reload
```

### Frontend Development

```bash
cd apps/frontend
pnpm dev
```

### Building for Production

```bash
# From root directory
pnpm build

# Backend (create deployment package)
cd apps/backend
pip freeze > requirements.txt

# Frontend (creates dist folder)
cd apps/frontend
pnpm build
```

## Troubleshooting

1. **"Cannot connect to server"**: Ensure the backend is running on port 8000
2. **"Invalid URL"**: Make sure you're using a Naver Webtoon URL
3. **Translation fails**: Check your OpenAI API key and rate limits
4. **CORS errors**: Ensure the frontend URL in backend .env matches your frontend URL

## Future Enhancements

1. Integrate with actual image generation/editing APIs for true text replacement
2. Add batch processing for multiple chapters
3. Implement caching to avoid re-translating the same images
4. Add user authentication and history
5. Support for other manhwa platforms
6. Advanced text detection and OCR for better accuracy

## License

This project is for educational purposes only. Please respect copyright laws and website terms of service when using this application.
