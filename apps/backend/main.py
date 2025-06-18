import os
import asyncio
import base64
from typing import List, Dict, Any, Tuple
from io import BytesIO
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
from openai import AsyncOpenAI
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import aiofiles
import cv2
import numpy as np
import easyocr

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Manhwa Translation API",
    description="API for translating Korean manhwa to English",
    version="1.0.0"
)

# Configure CORS
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
frontend_urls = [frontend_url, "http://localhost:5174", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_urls,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not set. Translation functionality will not work.")
    openai_client = None
else:
    openai_client = AsyncOpenAI(api_key=openai_api_key)

# Initialize OCR reader for Korean text
print("Initializing Korean OCR reader...")
try:
    ocr_reader = easyocr.Reader(['ko', 'en'], gpu=False)
    print("OCR reader initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize OCR reader: {e}")
    ocr_reader = None

# Request/Response models
class TranslateRequest(BaseModel):
    url: HttpUrl

class TranslateResponse(BaseModel):
    translated_images: List[str]
    original_count: int
    success: bool
    message: str = ""

class DownloadRequest(BaseModel):
    images: List[str]  # Base64 encoded images

# AI Prompt Template for Translation
TRANSLATION_PROMPT_TEMPLATE = """You are an expert image editor and translator specializing in Korean manhwa/webtoon translation. Your task is to analyze the provided manhwa panel image and create a new version with all Korean text translated to English.

CRITICAL INSTRUCTIONS:
1. Identify ALL Korean text in the image, including:
   - Speech bubbles
   - Thought bubbles
   - Sound effects (onomatopoeia)
   - Narrative boxes
   - Signs or labels
   - Any other text elements

2. Translate the Korean text to natural, contextually appropriate English that preserves:
   - The original meaning and tone
   - Character voice and personality
   - Emotional nuance
   - Appropriate formality levels

3. Generate a NEW image where:
   - All Korean text is REPLACED with the English translation
   - The English text EXACTLY matches the original's:
     * Font style and weight
     * Font size and scaling
     * Text color and opacity
     * Text effects (stroke, shadow, glow, gradient)
     * Text orientation and alignment
     * Integration with speech bubbles or text areas
   - The text should look like it was originally drawn in English
   - Preserve all artwork, characters, backgrounds exactly as they are
   - Maintain the same image dimensions and quality

4. Special considerations:
   - For sound effects, translate to appropriate English onomatopoeia
   - Preserve any stylistic text effects (wavy, bold, italic, etc.)
   - Ensure text fits naturally within speech bubbles without overflow
   - Match the artistic style of the original text presentation

Please process the image and return a new version with all Korean text professionally replaced with styled English translations that seamlessly blend with the original artwork.

[IMAGE_DATA_OR_URL_HERE]"""

async def scrape_webtoon_images(url: str) -> List[str]:
    """Scrape all image URLs from a Naver Webtoon page."""
    try:
        # Fetch the page
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the webtoon viewer div
        viewer_div = soup.find('div', class_='wt_viewer')
        if not viewer_div:
            raise ValueError("Could not find webtoon viewer div")
        
        # Extract all image URLs
        image_urls = []
        for img in viewer_div.find_all('img'):
            src = img.get('src')
            if src:
                # Ensure absolute URL
                if src.startswith('//'):
                    src = 'https:' + src
                elif src.startswith('/'):
                    src = 'https://comic.naver.com' + src
                image_urls.append(src)
        
        if not image_urls:
            raise ValueError("No images found in the webtoon viewer")
        
        return image_urls
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webtoon page: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scraping images: {str(e)}")

async def download_image(url: str) -> bytes:
    """Download an image from URL and return as bytes."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Referer': 'https://comic.naver.com/'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download image: {str(e)}")

async def translate_text_with_openai(korean_text: str) -> str:
    """Translate Korean text to English using OpenAI."""
    try:
        if not openai_client or not korean_text.strip():
            return korean_text
        
        response = await openai_client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Korean to English translator specializing in manhwa/webtoon content. Translate the given Korean text to natural English while preserving the tone, emotion, and character voice. For sound effects (onomatopoeia), translate to appropriate English equivalents. Keep translations concise to fit in speech bubbles."
                },
                {
                    "role": "user",
                    "content": f"Translate this Korean text to English: {korean_text}"
                }
            ],
            max_tokens=3000,
            temperature=0.3
        )
        
        translation = response.choices[0].message.content.strip()
        # Remove quotes if the AI added them
        translation = re.sub(r'^["\'](.*)["\']$', r'\1', translation)
        return translation
        
    except Exception as e:
        print(f"Translation error for '{korean_text}': {e}")
        return korean_text

def detect_and_extract_text(image_bytes: bytes) -> List[Dict]:
    """Detect Korean text in image and return text regions with coordinates."""
    try:
        if not ocr_reader:
            return []
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return []
        
        # Use EasyOCR to detect text
        results = ocr_reader.readtext(image)
        
        text_regions = []
        for (bbox, text, confidence) in results:
            # Only process if confidence is reasonable and text contains Korean characters
            if confidence > 0.3 and any('\u3130' <= char <= '\u318F' or '\uAC00' <= char <= '\uD7A3' for char in text):
                # Convert bbox to simple rectangle coordinates
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                
                text_regions.append({
                    'text': text.strip(),
                    'bbox': {
                        'x': int(min(x_coords)),
                        'y': int(min(y_coords)),
                        'width': int(max(x_coords) - min(x_coords)),
                        'height': int(max(y_coords) - min(y_coords))
                    },
                    'confidence': confidence
                })
        
        return text_regions
        
    except Exception as e:
        print(f"Text detection error: {e}")
        return []

def create_text_mask(image_size: Tuple[int, int], text_regions: List[Dict]) -> np.ndarray:
    """Create a mask for text regions to be inpainted."""
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    for region in text_regions:
        bbox = region['bbox']
        # Expand the bounding box slightly to ensure we cover all text
        padding = 5
        x1 = max(0, bbox['x'] - padding)
        y1 = max(0, bbox['y'] - padding)
        x2 = min(image_size[0], bbox['x'] + bbox['width'] + padding)
        y2 = min(image_size[1], bbox['y'] + bbox['height'] + padding)
        
        mask[y1:y2, x1:x2] = 255
    
    return mask

def inpaint_text_regions(image_bytes: bytes, text_regions: List[Dict]) -> bytes:
    """Remove text from image using inpainting."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or not text_regions:
            return image_bytes
        
        # Create mask for text regions
        mask = create_text_mask((image.shape[1], image.shape[0]), text_regions)
        
        # Apply inpainting to remove text
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', inpainted, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return buffer.tobytes()
        
    except Exception as e:
        print(f"Inpainting error: {e}")
        return image_bytes

def add_english_text(image_bytes: bytes, translations: List[Dict]) -> bytes:
    """Add English text to the inpainted image."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_bytes))
        draw = ImageDraw.Draw(image)
        
        for translation in translations:
            bbox = translation['bbox']
            english_text = translation['english_text']
            
            if not english_text:
                continue
            
            # Calculate font size based on bounding box height
            font_size = max(12, min(40, int(bbox['height'] * 0.7)))
            
            try:
                # Try to use a nice font, fall back to default
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
                except:
                    font = ImageFont.load_default()
            
            # Calculate text position (center in bounding box)
            bbox_info = draw.textbbox((0, 0), english_text, font=font)
            text_width = bbox_info[2] - bbox_info[0]
            text_height = bbox_info[3] - bbox_info[1]
            
            x = bbox['x'] + (bbox['width'] - text_width) // 2
            y = bbox['y'] + (bbox['height'] - text_height) // 2
            
            # Draw text with outline for better visibility
            outline_color = "black"
            text_color = "white"
            
            # Draw outline
            for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-1,0), (1,0), (0,-1), (0,1)]:
                draw.text((x+dx, y+dy), english_text, font=font, fill=outline_color)
            
            # Draw main text
            draw.text((x, y), english_text, font=font, fill=text_color)
        
        # Convert back to bytes
        output_buffer = BytesIO()
        image.save(output_buffer, format='JPEG', quality=95)
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
        
    except Exception as e:
        print(f"Text rendering error: {e}")
        return image_bytes

async def translate_image_with_ai(image_bytes: bytes, image_index: int) -> str:
    """Detect Korean text, translate it, and replace with English text."""
    try:
        print(f"  🔍 Step 1: OCR text detection for image {image_index + 1}")
        text_regions = detect_and_extract_text(image_bytes)
        
        if not text_regions:
            print(f"  ❌ No Korean text detected in image {image_index + 1}")
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
        
        print(f"  ✅ Found {len(text_regions)} text regions in image {image_index + 1}")
        for i, region in enumerate(text_regions):
            print(f"    Region {i+1}: '{region['text']}' (confidence: {region['confidence']:.2f})")
        
        print(f"  🌐 Step 2: Translating {len(text_regions)} text regions...")
        translations = []
        for i, region in enumerate(text_regions):
            korean_text = region['text']
            print(f"    🔄 Translating region {i+1}: '{korean_text}'")
            
            english_text = await translate_text_with_openai(korean_text)
            print(f"    ✅ Translation {i+1}: '{korean_text}' → '{english_text}'")
            
            translations.append({
                'bbox': region['bbox'],
                'korean_text': korean_text,
                'english_text': english_text,
                'confidence': region['confidence']
            })
        
        print(f"  🎨 Step 3: Removing original Korean text using inpainting...")
        inpainted_bytes = inpaint_text_regions(image_bytes, text_regions)
        print(f"  ✅ Text removal completed for image {image_index + 1}")
        
        print(f"  ✍️  Step 4: Adding English text to image {image_index + 1}...")
        final_image_bytes = add_english_text(inpainted_bytes, translations)
        print(f"  ✅ English text rendering completed for image {image_index + 1}")
        
        # Convert to base64 and return
        base64_image = base64.b64encode(final_image_bytes).decode('utf-8')
        print(f"  🎉 Successfully processed image {image_index + 1} with {len(translations)} translations!")
        
        return f"data:image/jpeg;base64,{base64_image}"
        
    except Exception as e:
        print(f"  ❌ Error processing image {image_index + 1}: {str(e)}")
        # Return original image on error
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_image}"

def combine_images_to_strip(base64_images: List[str]) -> bytes:
    """Combine multiple base64 images into a single vertical strip."""
    try:
        print(f"📥 Starting to combine {len(base64_images)} images into strip...")
        images = []
        total_width = 0
        total_height = 0
        
        # Decode all images and get dimensions
        for i, img_data in enumerate(base64_images):
            try:
                print(f"  Processing image {i+1}/{len(base64_images)} for strip...")
                
                # Remove data URL prefix if present
                if img_data.startswith('data:image'):
                    img_data = img_data.split(',')[1]
                
                # Decode base64 image
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))
                
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                print(f"    Image {i+1}: {img.width}x{img.height} pixels")
                images.append(img)
                total_width = max(total_width, img.width)
                total_height += img.height
                
            except Exception as e:
                print(f"    ❌ Error processing image {i+1}: {str(e)}")
                continue
        
        if not images:
            raise ValueError("No valid images could be processed")
        
        print(f"  📐 Creating combined strip: {total_width}x{total_height} pixels")
        
        # Create a new image with the combined dimensions
        combined_image = Image.new('RGB', (total_width, total_height), color='white')
        
        # Paste each image vertically
        current_y = 0
        for i, img in enumerate(images):
            # Center the image horizontally if it's narrower than the max width
            x_offset = (total_width - img.width) // 2
            print(f"    Placing image {i+1} at position y={current_y}")
            combined_image.paste(img, (x_offset, current_y))
            current_y += img.height
        
        # Save to bytes
        print(f"  💾 Saving combined strip as JPEG...")
        img_buffer = BytesIO()
        combined_image.save(img_buffer, format='JPEG', quality=95)
        img_buffer.seek(0)
        
        result_bytes = img_buffer.getvalue()
        print(f"  ✅ Successfully created strip: {len(result_bytes)} bytes")
        
        return result_bytes
        
    except Exception as e:
        print(f"❌ Error combining images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to combine images: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Manhwa Translation API",
        "version": "1.0.0",
        "endpoints": {
            "/api/translate-manhwa": "POST - Translate a manhwa from URL"
        }
    }

@app.post("/api/translate-manhwa", response_model=TranslateResponse)
async def translate_manhwa(request: TranslateRequest):
    """Main endpoint to translate a manhwa from a Naver Webtoon URL."""
    try:
        # Validate URL is from Naver Webtoon
        url_str = str(request.url)
        if "comic.naver.com" not in url_str:
            raise HTTPException(
                status_code=400, 
                detail="Invalid URL. Please provide a Naver Webtoon URL."
            )
        
        # Scrape image URLs
        print(f"Scraping images from: {url_str}")
        image_urls = await scrape_webtoon_images(url_str)
        print(f"Found {len(image_urls)} images")
        
        # Limit to first 10 images for testing
        if len(image_urls) > 30:
            print(f"Limiting to first 10 images for testing (found {len(image_urls)} total)")
            image_urls = image_urls[:30]
        else:
            print(f"Processing all {len(image_urls)} images")
        
        # Process images concurrently with limited concurrency
        translated_images = []
        semaphore = asyncio.Semaphore(2)  # Limit concurrent requests for better logging visibility
        
        async def process_image(url: str, index: int) -> str:
            async with semaphore:
                print(f"\n=== STARTING IMAGE {index + 1}/{len(image_urls)} ===")
                print(f"Image URL: {url[:100]}..." if len(url) > 100 else f"Image URL: {url}")
                
                print(f"[{index + 1}/{len(image_urls)}] Downloading image...")
                image_bytes = await download_image(url)
                print(f"[{index + 1}/{len(image_urls)}] Downloaded {len(image_bytes)} bytes")
                
                print(f"[{index + 1}/{len(image_urls)}] Starting translation process...")
                translated = await translate_image_with_ai(image_bytes, index)
                print(f"[{index + 1}/{len(image_urls)}] ✅ COMPLETED IMAGE {index + 1}")
                
                return translated
        
        # Create tasks for all images
        tasks = [
            process_image(url, i) 
            for i, url in enumerate(image_urls)
        ]
        
        # Wait for all translations to complete
        translated_images = await asyncio.gather(*tasks)
        
        return TranslateResponse(
            translated_images=translated_images,
            original_count=len(image_urls),
            success=True,
            message=f"Successfully processed {len(translated_images)} images"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/api/download-strip")
async def download_combined_strip(request: DownloadRequest):
    """Combine translated images into a single vertical strip and return as downloadable file."""
    try:
        print(f"\n🎬 === STARTING DOWNLOAD REQUEST ===")
        print(f"📊 Received {len(request.images)} images for download")
        
        if not request.images:
            print("❌ No images provided in request")
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Combine images into a single strip
        print(f"🔧 Combining {len(request.images)} images into vertical strip...")
        combined_image_bytes = combine_images_to_strip(request.images)
        
        print(f"📦 Returning download file: {len(combined_image_bytes)} bytes")
        print(f"✅ === DOWNLOAD REQUEST COMPLETED ===\n")
        
        # Return as downloadable file
        return Response(
            content=combined_image_bytes,
            media_type="image/jpeg",
            headers={
                "Content-Disposition": "attachment; filename=manhwa_translated_strip.jpg"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "manhwa-translator-backend"}

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    ) 