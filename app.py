import easyocr
import cv2
import numpy as np
import pandas as pd
import base64
# We will use this to convert PDF to an image array
from pdf2image import convert_from_bytes 
from flask import Flask, request, render_template_string
import sys 
import os

# --- Flask App Setup ---
app = Flask(__name__)

# --- Configuration and Initialization ---
# Global variable for the OCR reader
reader = None

def load_ocr_reader():
    """
    Initializes the EasyOCR reader once when the server starts.
    ATTEMPTS TO USE GPU FOR MAXIMUM SPEED.
    """
    global reader
    print("Initializing EasyOCR reader (English, Attempting GPU Mode)...")
    try:
        # **MODIFIED LINE:** Set gpu=True to enable GPU acceleration
        # EasyOCR will automatically check for a compatible CUDA installation.
        reader = easyocr.Reader(['en'], gpu=True) 
        print("✅ EasyOCR initialized successfully using GPU model (CUDA).")
    except Exception as e:
        print(f"❌ FATAL: Error initializing EasyOCR with GPU. Attempting CPU fallback.")
        print(f"Error details: {e}")
        
        # Fallback to non-quantized CPU model if GPU initialization fails
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            print("⚠️ EasyOCR initialized with default CPU settings (Fallback). Performance will be slow.")
        except Exception as e_f:
            print(f"❌ FATAL: EasyOCR failed even with CPU fallback. App will not function. {e_f}")
            reader = None

# Load the model immediately when the script executes
with app.app_context():
    load_ocr_reader()

# --- Core OCR and Data Extraction Logic (No changes needed here) ---
def get_center(bbox):
    """Helper function to find the center point of a bounding box."""
    x_coords = [p[0] for p in bbox]
    y_coords = [p[1] for p in bbox]
    return (sum(x_coords) / 4, sum(y_coords) / 4)

def process_ocr(image_array):
    """
    Performs OCR, sorts results, and structures the data into a DataFrame.
    Returns (DataFrame, image_with_boxes_base64)
    """
    if image_array is None:
        return pd.DataFrame([["Error", "No valid image provided for OCR."]]), None

    # EasyOCR expects RGB
    img_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    
    # Run OCR
    # detail=1 returns bounding box, text, and confidence
    results = reader.readtext(img_rgb, detail=1) 
    
    # Sort the results primarily by Y-coordinate (row) then by X-coordinate (column)
    sorted_results = sorted(results, key=lambda r: (get_center(r[0])[1], get_center(r[0])[0]))

    # Group into logical rows
    ROW_TOLERANCE = 20 # Max vertical distance allowed between texts in the same row
    grouped_rows = []
    
    if sorted_results:
        current_row = [sorted_results[0]]
        baseline_y = get_center(sorted_results[0][0])[1]

        for i in range(1, len(sorted_results)):
            r = sorted_results[i]
            center_y = get_center(r[0])[1]

            if abs(center_y - baseline_y) < ROW_TOLERANCE:
                current_row.append(r)
            else:
                # Sort the completed row by X position before appending
                current_row.sort(key=lambda item: get_center(item[0])[0])
                grouped_rows.append(current_row)
                
                # Start a new row
                current_row = [r]
                baseline_y = center_y

        if current_row:
            current_row.sort(key=lambda item: get_center(item[0])[0])
            grouped_rows.append(current_row)
    
    # Convert grouped rows to DataFrame
    extracted_data = [[item[1] for item in row] for row in grouped_rows]
    df = pd.DataFrame(extracted_data)

    # Create visualization image with bounding boxes
    img_boxes = image_array.copy()
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        # Draw box and text
        cv2.rectangle(img_boxes, tl, br, (0, 255, 0), 2) 
        cv2.putText(img_boxes, text, (tl[0], tl[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Convert image with boxes back to Base64 for HTML display
    # Convert BGR (OpenCV) back to RGB before encoding PNG 
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_boxes, cv2.COLOR_BGR2RGB))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return df, img_base64

def handle_image_upload(file):
    """
    Handles both image and PDF file uploads.
    """
    try:
        if file.filename.lower().endswith('.pdf'):
            # Process PDF: Reduced DPI to 150 for faster CPU conversion and lower memory use
            pdf_bytes = file.read()
            if reader is None:
                    return "Error: OCR Reader not initialized.", None

            # Use pdf2image to convert the first page of the PDF to a PIL image
            pages = convert_from_bytes(pdf_bytes, 
                                        first_page=1, 
                                        last_page=1, 
                                        dpi=150,
                                        fmt='jpeg') # Use jpeg format for lower memory footprint
            if pages:
                source_image = pages[0]
                img_array = np.array(source_image.convert('RGB'))
                # Convert PIL RGB to OpenCV BGR format
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
            else:
                return "Error: Could not process PDF page.", None
        else:
            # Handle standard image file upload (JPG/PNG/etc)
            image_bytes = file.read()
            img_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
        if img_array is None:
             return "Error: File type not supported or corrupted file.", None
             
        # Optional: Resize large images to reduce OCR time and memory
        max_dim = 1500 # Max dimension in pixels (width or height)
        h, w = img_array.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            print(f"Resized image from {w}x{h} to {img_array.shape[1]}x{img_array.shape[0]}")
        
        return "File loaded successfully.", img_array

    except Exception as e:
        return f"Error loading file: {e}", None


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    df_html = None
    image_b64 = None
    status = "Waiting for document upload."
    image_array = None

    if request.method == 'POST':
        # 1. Handle standard file upload (Image/PDF)
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            status, image_array = handle_image_upload(file)
        else:
            status = "No file selected for upload."
            
        # Run OCR if a valid image array was obtained and reader is available
        if image_array is not None and reader is not None:
            # Update status for front-end feedback
            status = "Processing OCR (GPU Accelerated: This should be fast)..."
            
            try:
                df, image_b64 = process_ocr(image_array)
                # Convert DataFrame to a styled HTML table
                df_html = df.to_html(classes='table w-full text-sm text-left text-gray-500')
                status = "✅ OCR Processing Complete!"
            except Exception as e:
                status = f"❌ An OCR processing error occurred: {e}"
        elif reader is None:
             status = "❌ FATAL: OCR Reader failed to initialize. Cannot process documents."
        elif image_array is None and status == "File loaded successfully.":
             status = "❌ Could not decode image file. Check the file format."


    # --- Embedded HTML Template (No changes needed here) ---
    HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OCR Scanner</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .container { max-width: 100%; padding: 0.5rem; }
        .table th, .table td { padding: 0.5rem; border: 1px solid #e5e7eb; }
    </style>
</head>
<body class="bg-gray-100 font-sans p-2 md:p-4">

    <div class="container mx-auto bg-white shadow-2xl rounded-xl p-4 md:p-8 my-4 md:my-8">
        <h1 class="text-3xl md:text-4xl font-extrabold text-indigo-700 mb-2">
            📄 EasyOCR Document Scanner
        </h1>
        <p class="text-indigo-500 mb-6 border-b pb-4">
            Upload an Image (JPG, PNG) or PDF to extract text.
        </p>

        <div class="grid grid-cols-1 gap-6">
            
            <div class="space-y-4">
                <h2 class="text-xl md:text-2xl font-bold text-gray-800 border-l-4 border-indigo-500 pl-3">1. Upload Document</h2>
                
                <form method="POST" enctype="multipart/form-data" class="space-y-4 p-4 border border-gray-200 rounded-xl shadow-md bg-white">
                    <label for="file" class="block text-md font-medium text-gray-700">Choose Image or PDF</label>
                    <input type="file" name="file" id="file" class="w-full text-gray-600 file:mr-4 file:py-3 file:px-6 file:rounded-full file:border-0 file:text-base file:font-semibold file:bg-indigo-50 file:text-indigo-700 hover:file:bg-indigo-100 cursor-pointer" accept=".jpg, .jpeg, .png, .pdf">
                    <button type="submit" class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-xl transition duration-300 shadow-lg">
                        🚀 Upload and Scan Document
                    </button>
                </form>
            </div>

            <div class="result-container bg-gray-50 p-4 rounded-xl shadow-inner border border-gray-200">
                <h2 class="text-xl md:text-2xl font-bold text-gray-800 border-l-4 border-red-500 pl-3">2. Extracted Data</h2>
                
                <p id="status_message" class="mt-4 mb-4 font-semibold p-3 rounded-lg text-sm 
                    {% if 'Error' in status or 'FATAL' in status or '❌' in status %}text-red-700 bg-red-100{% elif 'Complete' in status or '✅' in status %}text-green-700 bg-green-100{% elif 'Processing' in status %}text-yellow-700 bg-yellow-100{% else %}text-indigo-700 bg-indigo-100{% endif %}">
                    {{ status }}
                </p>

                {% if image_b64 %}
                    <h3 class="text-lg md:text-xl font-medium mt-6 mb-3 text-gray-700">OCR Visualization (Bounding Boxes)</h3>
                    <img src="data:image/png;base64,{{ image_b64 }}" alt="OCR Bounding Boxes" class="w-full h-auto rounded-lg shadow-md mb-6 border-2 border-gray-300">
                    
                    <h3 class="text-lg md:text-xl font-medium mt-6 mb-3 text-gray-700">Structured Output</h3>
                    <div class="overflow-x-auto p-2 bg-white rounded-lg border">
                        {{ df_html|safe }}
                    </div>
                {% endif %}
            </div>

        </div>
    </div>
</body>
</html>
    """
    
    return render_template_string(HTML_TEMPLATE, 
                                  status=status, 
                                  image_b64=image_b64, 
                                  df_html=df_html)

if __name__ == '__main__':
    # Set host to 0.0.0.0 for external access in a container/network (necessary for mobile testing)
    app.run(host='0.0.0.0', port=5000, debug=True)