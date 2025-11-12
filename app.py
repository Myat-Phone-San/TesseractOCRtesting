import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
import re
import pytesseract
import os
from io import BytesIO

# --- Configuration for Streamlit Cloud and Language Data ---
TESSERACT_LANGUAGES = 'eng+mya'

# CRITICAL FIX FOR CLOUD DEPLOYMENT:
# Set TESSDATA_PREFIX environment variable. 
# This tells Tesseract to look for the traineddata files in the 'tessdata' folder 
# relative to the current working directory (the root of your Streamlit repo).
if os.path.exists("tessdata"):
    # This block executes successfully when the folder is committed to GitHub
    os.environ['TESSDATA_PREFIX'] = 'tessdata'
    # Remove the st.info() for a cleaner deployed app
else:
    # If this warning shows, it means 'tessdata' is missing from the GitHub repo.
    st.error("The 'tessdata' folder is missing from the GitHub repository. Myanmar language OCR (mya) will fail. Please commit 'tessdata/mya.traineddata' to your repo.")

# Set the page configuration early
st.set_page_config(
    page_title="Document OCR Extractor (Eng + Myanmar)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) ---
# [x_min, y_min, x_max, y_max] for targeted extraction
TARGET_FIELD_REGIONS = {
    "Applicant Name": [50, 120, 480, 200], 
    "Documentary Credit No.": [480, 120, 680, 200],
    "Original Credit Amount": [680, 120, 930, 200],
    "Contact Person / Tel": [50, 200, 480, 300],
    "Beneficiary Name": [50, 200, 480, 400],
}

# --- Core Extraction Logic (Targeted by Region) ---
def extract_fields_by_region(image_array):
    """ Extracts text for the five specified fields using targeted regions. """
    kv_data = {key: '-' for key in TARGET_FIELD_REGIONS.keys()}
    H, W, _ = image_array.shape
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)

    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in TARGET_FIELD_REGIONS.items():
        # 1. Denormalize coordinates
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)

        # 2. Crop and run Tesseract
        cropped_img = image_array[y_min:y_max, x_min:x_max]
        if cropped_img.size == 0: continue
        
        text_raw = pytesseract.image_to_string(
            cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), 
            lang=TESSERACT_LANGUAGES
        ).strip()
        
        extracted_value = '-'
        if text_raw:
            # Simplified value extraction logic
            lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
            
            key_index = -1
            for i, line in enumerate(lines):
                if key.split(' ')[0].lower() in line.lower(): 
                    key_index = i
                    break

            if key_index != -1:
                value_lines = lines[key_index + 1:]
                if value_lines:
                    extracted_value = " ".join(value_lines)
                elif len(lines) == 1 and key.lower() in lines[0].lower():
                    # Same line key-value handling
                    key_end_index = lines[0].lower().find(key.lower()) + len(key)
                    value_on_same_line = lines[0][key_end_index:].strip().replace('—', '').replace('-', '').replace(':', '').replace('.', '').strip()
                    if value_on_same_line:
                        extracted_value = value_on_same_line
            
            if extracted_value == '-' and len(lines) > 0 and key.split(' ')[0].lower() not in text_raw.lower():
                extracted_value = " ".join(lines) 

        # 5. Post-process specific fields (e.g., currency)
        if key == "Original Credit Amount" and extracted_value != '-':
            amount_match = re.search(r'[\d\.\,]+', extracted_value.replace(' ', ''))
            if amount_match:
                amount_str = amount_match.group(0).replace(',', '').strip()
                try:
                    extracted_value = f"EUR {float(amount_str):,.2f}"
                except ValueError:
                    extracted_value = f"EUR {amount_str}"
            elif 'EUR' not in extracted_value.upper():
                 extracted_value = f"EUR {extracted_value}"
                 
        elif extracted_value != '-':
            extracted_value = extracted_value.replace('\n', ' ').strip()
            
        if extracted_value and extracted_value != '-':
            kv_data[key] = extracted_value.strip()
        else:
             kv_data[key] = '-'

        # Draw the box for visualization (Red border)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    img_with_boxes = Image.fromarray(img_boxes)
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)
    
    return df_kv_pairs, img_with_boxes

# --- Full Text Extraction (Non-Structured) ---
def extract_full_text(image_array):
    """ Runs Tesseract on the entire image to get non-structured text. """
    with st.spinner("Extracting full page text (Non-Structured)..."):
        full_text = pytesseract.image_to_string(
            cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            lang=TESSERACT_LANGUAGES
        ).strip()
    return full_text

# --- Utility Functions ---

def handle_file_upload(uploaded_file):
    """Handles file uploads, converting them to an OpenCV BGR array."""
    file_type = uploaded_file.type
    try:
        file_bytes = uploaded_file.read()
        if 'pdf' in file_type:
            with st.spinner("Converting PDF page 1 to image (150 DPI)..."):
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                if doc.page_count == 0:
                    st.error("Could not process PDF. The document is empty or unreadable.")
                    return None
                page = doc.load_page(0)
                DPI = 150
                zoom_factor = DPI / 72
                matrix = fitz.Matrix(zoom_factor, zoom_factor)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img_array_bgr = cv2.cvtColor(img_array_rgb, cv2.COLOR_RGB2BGR) 
                doc.close()
                return img_array_bgr
        else:
            img_array_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array_bgr
    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    
    mime = 'text/plain' # Default mime type
    
    if is_dataframe:
        if file_format == 'csv':
            data_out = data.to_csv(index=False).encode('utf-8')
            mime = 'text/csv'
        else: # txt or doc for DataFrame
            data_out = data.to_string(index=False, header=True).encode('utf-8')
    else: # txt or doc for raw text
        data_out = data.encode('utf-8')
        
    final_name = f'{file_name_base}.{file_format}'
        
    st.download_button(
        label=label,
        data=data_out,
        file_name=final_name,
        mime=mime,
        help=help_text
    )

# --- Streamlit Application Layout ---
def main():
    
    st.title("🎯Document OCR Extractor (English & Myanmar)")
    st.markdown("This tool uses **Tesseract OCR ** to extract specific fields and the full page text from a document.")

    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose a Document File (Image or PDF)",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed."
    )
    st.markdown("---")

    image_array_bgr = None
    if uploaded_file is not None:
        st.info(f"File **'{uploaded_file.name}'** uploaded. Starting file conversion...")
        image_array_bgr = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---
    if image_array_bgr is not None:
        
        st.subheader("2. OCR Processing and Result Formats")
        
        # Run both extractions
        df_kv_pairs, img_with_boxes = extract_fields_by_region(image_array_bgr)
        full_text = extract_full_text(image_array_bgr)

        col_img, col_data_tabs = st.columns([1, 2])
        
        with col_img:
            st.markdown("### 🖼️ OCR Visualization")
            st.image(img_with_boxes, caption="Targeted extraction regions (Red Boxes)", use_column_width=True)
            
        with col_data_tabs:
            # Create two tabs: one for Structured, one for Non-Structured
            tab_structured, tab_non_structured = st.tabs(["📄 Structured Table", "📋 All Non-Structured Text"])

            with tab_structured:
                st.markdown("### 🔑 Extracted Key-Value Pairs")
                st.dataframe(
                    df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], 
                    use_container_width=True, 
                    hide_index=True
                )
                
                st.markdown("#### Download Options (Table)")
                col_csv, col_txt, col_word = st.columns(3)
                with col_csv:
                    get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'structured_key_value_pairs')
                with col_txt:
                    get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'structured_key_value_pairs')
                with col_word:
                    get_download_button(df_kv_pairs, True, 'doc', "📥 Download DOC", 'structured_key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
            
            with tab_non_structured:
                st.markdown("### Full Extracted Text (Sorted by Read Order)")
                st.text_area(
                    label="Non-Structured Text",
                    value=full_text,
                    height=400,
                    help="This is the raw text output from Tesseract across the entire image."
                )
                
                st.markdown("#### Download Options (Full Text)")
                col_txt_full, col_word_full, _ = st.columns(3)
                with col_txt_full:
                    get_download_button(full_text, False, 'txt', "📥 Download TXT", 'full_extracted_text')
                with col_word_full:
                    get_download_button(full_text, False, 'doc', "📥 Download DOC", 'full_extracted_text', help_text="Saves the full text as a text file with a .doc extension.")
                
        st.markdown("---")
       
if __name__ == '__main__':
    main()
