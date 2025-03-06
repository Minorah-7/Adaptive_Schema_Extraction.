import streamlit as st
import cv2
import numpy as np
import pytesseract
import pandas as pd
import tempfile
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Input Folder Containing Cropped Images
input_folder = r"C:\Users\minor\OneDrive\Desktop\output_ed"  # Update this path

# Define ROIs
horizontal_rois = [
    (415, 215, 55, 25), (460, 185, 55, 25), (518, 213, 55, 25),
    (590, 215, 55, 25), (410, 630, 55, 25), (450, 1023, 55, 25), (310, 630, 55, 25),
    
    (1070, 215, 55, 25), (1130, 240, 55, 25), (1170, 215, 55, 25),
    (1000, 215, 55, 25), (1270, 630, 55, 25), (1135, 1023, 55, 25), (1170, 630, 55, 25),
    
    (1665, 215, 55, 25), (1799, 240, 55, 25), (1750, 215, 55, 25),
    (1840, 215, 55, 25), (1945, 630, 55, 25), (1800, 1023, 55, 25), (1840, 630, 55, 25)
]
vertical_rois = [
    (163, 380, 25, 55), (165, 475, 25, 55), (75, 720, 25, 55),
    (440, 700, 25, 55), (165, 850, 25, 55), (165, 940, 25, 55), (165, 990, 25, 55),
    
    (1363, 380, 25, 55), (1363, 475, 25, 55), (1140, 680, 25, 55),
    (1365, 850, 25, 55), (1365, 940, 25, 55), (1365, 1010, 25, 55),

    (2085, 380, 25, 55), (2085, 475, 25, 55), (1805, 700, 25, 55),
    (2085, 850, 25, 55), (2085, 940, 25, 55), (2085, 1010, 25, 55)
]

# Function to extract text from ROIs
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted_values = []

    # Extract text from horizontal ROIs
    for (x, y, w, h) in horizontal_rois:
        roi = gray[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6').strip()
        extracted_values.append(text)

    # Extract text from vertical ROIs (rotate before OCR)
    for (x, y, w, h) in vertical_rois:
        roi = gray[y:y+h, x:x+w]
        roi_rotated = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        text = pytesseract.image_to_string(roi_rotated, config='--psm 6').strip()
        extracted_values.append(text)

    return extracted_values

# Streamlit UI
st.title("Adaptive Schema Extraction for Manufacturing Drawings")

uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    df = pd.DataFrame(columns=["Image"] + [f"Field {i+1}" for i in range(len(horizontal_rois) + len(vertical_rois))])

    for uploaded_file in uploaded_files:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Extract text
        extracted_texts = extract_text(image)

        # Store in DataFrame
        df.loc[len(df)] = [uploaded_file.name] + extracted_texts

        # Draw ROIs on the image
        image_with_rois = image.copy()
        for (x, y, w, h) in horizontal_rois + vertical_rois:
            cv2.rectangle(image_with_rois, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Convert to PIL format for display
        st.image(cv2.cvtColor(image_with_rois, cv2.COLOR_BGR2RGB), caption="Detected ROIs", use_column_width=True)

    # Show extracted text
    st.write("### Extracted Data")
    st.dataframe(df)

    # Download as Excel
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    df.to_excel(output_file.name, index=False)

    with open(output_file.name, "rb") as f:
        st.download_button("📥 Download Extracted Data", f, "extracted_text.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

