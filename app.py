import cv2
import numpy as np
import pandas as pd
import streamlit as st
import fitz # PyMuPDF
from PIL import Image
from io import BytesIO
import re 
import pytesseract

# --- Configuration and Initialization ---
# Removed TESSERACT_PATH and configure_tesseract() function.
# Tesseract will be automatically found on the system path after 
# being installed via packages.txt in the cloud environment.

# Set the page configuration early
st.set_page_config(
    page_title="Document OCR Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Define Normalized Region Coordinates (0-1000 scale) ---
# These coordinates define the rectangular areas containing BOTH the key and the value.
# Normalized to 0-1000 for image independent scaling
# [x_min, y_min, x_max, y_max]

TARGET_FIELD_REGIONS = {
    "Applicant Name": [50, 120, 480, 200], # Captures Key and Value below it
    "Documentary Credit No.": [480, 120, 680, 200],
    "Original Credit Amount": [680, 120, 930, 200],
    "Contact Person / Tel": [50, 200, 480, 300], # Captures Key and Value below it
    # Adjusted Y-min slightly for Beneficiary to prevent overlap with Contact Person
    "Beneficiary Name": [50, 300, 480, 400], 
}

# --- Core Extraction Logic (Targeted by Region) ---

def extract_fields_by_region(image_array):
    """
    Extracts text for the five specified fields by cropping the image 
    to predefined normalized regions and running Tesseract on each.
    """
    kv_data = {key: '-' for key in TARGET_FIELD_REGIONS.keys()}
    H, W, _ = image_array.shape
    
    # Placeholder for drawing boxes
    img_boxes = cv2.cvtColor(image_array.copy(), cv2.COLOR_BGR2RGB)
    
    for key, (x_min_norm, y_min_norm, x_max_norm, y_max_norm) in TARGET_FIELD_REGIONS.items():
        
        # 1. Denormalize coordinates to actual pixel values
        x_min = int(x_min_norm * W / 1000)
        y_min = int(y_min_norm * H / 1000)
        x_max = int(x_max_norm * W / 1000)
        y_max = int(y_max_norm * H / 1000)
        
        # 2. Crop the image to the target region
        cropped_img = image_array[y_min:y_max, x_min:x_max]
        
        if cropped_img.size == 0:
            continue

        # 3. Run Tesseract on the cropped image (English only)
        # Using lang='eng' as requested.
        text_raw = pytesseract.image_to_string(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), lang='eng').strip()
        
        # 4. Process the extracted text
        extracted_value = '-'
        
        if text_raw:
            # Clean up common OCR artifacts and split lines
            lines = [line.strip() for line in text_raw.split('\n') if line.strip()]
            
            # Simple heuristic to find the value after the key label
            key_index = -1
            # Use the most significant part of the key for matching
            key_match_str = key.split(' ')[0].split('/')[0] 
            
            for i, line in enumerate(lines):
                # Search for the key match string robustly
                if re.search(r'\b' + re.escape(key_match_str) + r'\b', line, re.IGNORECASE):
                    key_index = i
                    break
            
            if key_index != -1:
                # The value is all subsequent text joined together
                value_lines = lines[key_index + 1:]
                
                # Handling single-line or multi-line key/value split
                if value_lines:
                    extracted_value = " ".join(value_lines)
                else:
                    # If key is found but no subsequent lines, try extracting value from the same line
                    key_clean = re.escape(key).replace(r'\ ', r'\\s*')
                    # Regex to capture content after the key, ignoring common delimiters
                    match = re.search(key_clean + r'[\s:-]*\s*(.*)', lines[key_index], re.IGNORECASE)
                    if match and match.group(1).strip():
                        extracted_value = match.group(1).strip()
            
            # Fallback: If no key found, assume the whole text is the value (common for fields where key is outside the box)
            if extracted_value == '-' and lines:
                 # Check if the text is clearly not the key itself
                if not any(re.search(r'\b' + re.escape(key.split(' ')[0]) + r'\b', line, re.IGNORECASE) for line in lines):
                    extracted_value = " ".join(lines)


        # 5. Post-process specific fields for better presentation/accuracy
        if key == "Original Credit Amount" and extracted_value != '-':
            # Attempt to clean and format the amount
            amount_match = re.search(r'[\d\.\,]+', extracted_value.replace(' ', ''))
            if amount_match:
                amount_str = amount_match.group(0).replace(',', '') # Remove commas used as thousands separator
                try:
                    amount_float = float(amount_str)
                    # Assuming EUR based on common form context if not explicitly captured
                    extracted_value = f"EUR {amount_float:,.2f}"
                except ValueError:
                    # If conversion fails, keep the original cleaned string
                    pass
            
        elif key in ["Applicant Name", "Contact Person / Tel", "Beneficiary Name"] and extracted_value != '-':
            # For multi-line fields, clean up extra spaces/line noise
            extracted_value = extracted_value.replace('\n', ' ').strip()
        
        # Final cleanup for values
        if extracted_value and extracted_value != '-':
            # Remove line drawing characters like | or - if they appear at ends
            extracted_value = extracted_value.strip('|.- ')
            kv_data[key] = extracted_value.strip()

        # Draw the box for visualization (Red border)
        cv2.rectangle(img_boxes, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # Convert the box-drawn image to PIL Image for Streamlit
    img_with_boxes = Image.fromarray(img_boxes)

    # Convert KV data to DataFrame
    kv_df_list = [{'Key Label (Form Text)': k, 'Extracted Value': v} for k, v in kv_data.items()]
    df_kv_pairs = pd.DataFrame(kv_df_list)

    return df_kv_pairs, img_with_boxes

# --- Utility Functions ---

def handle_file_upload(uploaded_file):
    """Handles file uploads, converting them to an OpenCV array."""
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
                
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) 
                doc.close()
                return img_array
            
        else: # Handle image files (jpg, png, etc.)
            img_array = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            return img_array

    except Exception as e:
        st.error(f"Error loading file. Check if it's a valid Image or non-encrypted PDF. Error details: {e}")
        return None

def get_download_button(data, is_dataframe, file_format, label, file_name_base, help_text=""):
    """Generates a common download button for different formats."""
    
    df = data
    if file_format == 'csv':
        data_out = df.to_csv(index=False).encode('utf-8')
        mime = 'text/csv'
        final_name = f'{file_name_base}.csv'
    else: # txt or doc
        data_out = df.to_string(index=False).encode('utf-8')
        mime = 'text/plain' if file_format == 'txt' else 'application/msword'
        final_name = f'{file_name_base}.{file_format}'
        
    st.download_button(
        label=label,
        data=data_out,
        file_name=final_name,
        mime=mime,
        help=help_text
    )

# --- Streamlit Application Layout (Simplified) ---

def main():
    
    st.title("🎯Document OCR Extractor (Tesseract)")
    st.markdown("This tool uses **Tesseract OCR** to extract fields from forms using predefined, targeted regions.")
    
    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Choose a Document File",
        type=['jpg', 'jpeg', 'png', 'pdf'],
        help="For multi-page PDFs, only the first page will be processed."
    )

    st.markdown("---")

    image_array = None
    if uploaded_file is not None:
        st.info(f"File **'{uploaded_file.name}'** uploaded. Starting file conversion...")
        image_array = handle_file_upload(uploaded_file)

    # --- OCR Processing and Results Display ---

    if image_array is not None:
        st.subheader("2. Extracted Results")
            
        # Run targeted extraction
        with st.spinner("Running targeted OCR on predefined regions..."):
            df_kv_pairs, img_with_boxes = extract_fields_by_region(image_array)

        # Display results in a two-column layout
        col_img, col_data = st.columns([1, 2])
        
        with col_img:
            st.markdown("### 🖼️ OCR Visualization (Targeted Regions)")
            st.image(img_with_boxes, caption="Targeted extraction regions (Red Boxes)", use_column_width=True)

        with col_data:
            st.markdown("### 🔑 Extracted Key-Value Pairs")
            
            # Displaying the custom, refined output
            st.dataframe(df_kv_pairs[['Key Label (Form Text)', 'Extracted Value']], use_container_width=True, hide_index=True)
            
            st.markdown("#### Download Options")
            col_csv, col_txt, col_word = st.columns(3)
            
            with col_csv:
                get_download_button(df_kv_pairs, True, 'csv', "📥 Download CSV", 'targeted_key_value_pairs')

            with col_txt:
                get_download_button(df_kv_pairs, True, 'txt', "📥 Download TXT", 'targeted_key_value_pairs')
                
            with col_word:
                get_download_button(df_kv_pairs, True, 'doc', "📥 Download DOC (Word)", 'targeted_key_value_pairs', help_text="Saves the table data as a text file with a .doc extension.")
                
    st.markdown("---")
 

if __name__ == '__main__':
    main()

