import streamlit as st
import cv2
import numpy as np
import pytesseract
import io
import os

# --- Configuration ---
# Note: If you encounter 'pytesseract.TesseractNotFoundError' locally, 
# you may need to set the command path.
# Example for Windows: pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# For Streamlit Cloud/Linux, the 'pytesseract' package (installed via requirements.txt) 
# usually finds the tesseract executable (installed via packages.txt).

def run_ocr(image_array):
    """
    Performs OCR on the given image array and returns the extracted text.
    Uses English language by default.
    """
    try:
        # Convert the image (optional: resize or preprocess if needed)
        # For simplicity, we use the raw image here.
        
        # Use pytesseract to extract text
        extracted_text = pytesseract.image_to_string(image_array)
        return extracted_text
    except Exception as e:
        # Log the error and return a message
        print(f"OCR Error: {e}")
        return f"Error during OCR processing: {e}"

# --- Streamlit UI Setup ---

st.set_page_config(
    page_title="Tesseract OCR Image Reader",
    layout="centered"
)

st.title("📄 Tesseract OCR Web App")
st.markdown("Upload an image containing text to extract the data.")

# File Uploader component
uploaded_file = st.file_uploader(
    "Choose an image file (PNG, JPG, JPEG)", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # 1. Read the image file as bytes
        image_bytes = uploaded_file.read()
        
        # 2. Convert bytes to a numpy array
        np_array = np.frombuffer(image_bytes, np.uint8)
        
        # 3. Decode the numpy array into an OpenCV image (color image)
        # cv2.IMREAD_COLOR loads the image in BGR format
        image_cv = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if image_cv is None:
             st.error("Could not decode image. Please check the file format.")
        else:
            # Convert BGR (OpenCV format) to RGB (Streamlit display format)
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

            st.subheader("Uploaded Image")
            st.image(image_rgb, caption=uploaded_file.name)
            
            st.markdown("---")
            
            with st.spinner('Running OCR...'):
                # Pass the RGB image array to the OCR function
                text_result = run_ocr(image_rgb)
            
            st.subheader("Extracted Text")
            st.code(text_result, language="text")

    except Exception as e:
        st.error(f"An unexpected error occurred during file processing: {e}")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses the power of **OpenCV (`cv2`)** for image decoding "
    "and **Pytesseract** for Optical Character Recognition (OCR)."
)
