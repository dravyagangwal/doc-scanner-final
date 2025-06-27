import os
import cv2
import numpy as np
import pytesseract
import re
import pandas as pd
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from pyzbar.pyzbar import decode
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img):
    """Preprocess image for English text extraction"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Sharpening
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(processed, -1, kernel_sharp)
    
    return sharpened

def preprocess_for_hindi(img):
    """Special preprocessing for Hindi text extraction"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoising with different parameters for Hindi
    denoised = cv2.fastNlMeansDenoising(gray, h=20, templateWindowSize=9, searchWindowSize=25)
    
    # Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(denoised)
    
    # Binarization with adaptive threshold
    thresh = cv2.adaptiveThreshold(contrast_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 31, 8)
    
    # Dilation to connect broken Hindi characters
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    return dilated

def extract_qr_data(image_path):
    """Extract data from QR code if present"""
    img = cv2.imread(image_path)
    decoded_objects = decode(img)
    if decoded_objects:
        return decoded_objects[0].data.decode('utf-8')
    return None

def process_aadhaar(image_path):
    """Process Aadhaar card image with separate preprocessing for English and Hindi"""
    try:
        img = cv2.imread(image_path)
        
        # Get QR code data first
        qr_data = extract_qr_data(image_path)
        
        # Preprocess for English
        english_preprocessed = preprocess_image(img)
        
        # Preprocess for Hindi (different parameters)
        hindi_preprocessed = preprocess_for_hindi(img)
        
        # OCR for English
        custom_eng_config = r'--oem 3 --psm 6'
        english_data = pytesseract.image_to_data(english_preprocessed, 
                                              output_type=pytesseract.Output.DICT, 
                                              lang='eng',
                                              config=custom_eng_config)
        english_text = pytesseract.image_to_string(english_preprocessed, lang='eng')
        
        # OCR for Hindi with combined language model
        custom_hin_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        hindi_data = pytesseract.image_to_data(hindi_preprocessed, 
                                             output_type=pytesseract.Output.DICT, 
                                             lang='hin+eng',
                                             config=custom_hin_config)
        hindi_text = pytesseract.image_to_string(hindi_preprocessed, lang='hin+eng')
        
        return english_data, hindi_data, english_text, hindi_text, qr_data
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None, None, None, None

def extract_hindi_name(hindi_text):
    """Improved Hindi name extraction with multiple patterns"""
    # Clean text by removing special characters and extra spaces
    cleaned_text = re.sub(r'[^\w\s]', '', hindi_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    name_patterns = [
        r'नाम[\s:\-]*([^\n]+)',
        r'नाम[^\w]*([^\n]+)',
        r'([^\n]+)\n.*आधार.*संख्या',
        r'([^\n]+)\n.*यूआईडी',
        r'Name in Hindi[\s:\-]*([^\n]+)'
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            extracted = match.group(1).strip()
            # Remove any remaining English text if mixed
            extracted = re.sub(r'[a-zA-Z]', '', extracted).strip()
            if extracted:
                return extracted
    
    return "नाम नहीं मिला"

def extract_hindi_address(hindi_text, eng_address):
    """Extract Hindi address if available, otherwise transliterate English"""
    # Try to find Hindi address patterns
    address_patterns = [
        r'पता[\s:\-]*([^\n]+(?:\n[^\n]+){0,4})',
        r'स्थायी पता[\s:\-]*([^\n]+(?:\n[^\n]+){0,4})',
        r'Address in Hindi[\s:\-]*([^\n]+(?:\n[^\n]+){0,4})'
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, hindi_text)
        if match:
            address_lines = [line.strip() for line in match.group(1).split('\n') if line.strip()]
            return '\n'.join(address_lines)
    
    # If no Hindi address found, transliterate English address
    if eng_address and eng_address != "Address not extracted":
        try:
            # Simple transliteration (for demonstration)
            transliterated = transliterate(eng_address, sanscript.ITRANS, sanscript.DEVANAGARI)
            return transliterated
        except:
            return "पता निकालने में असमर्थ"
    
    return "पता नहीं मिला"

def extract_aadhaar_details(english_data, hindi_data, english_text, hindi_text, qr_data):
    """Enhanced field extraction with better Hindi support"""
    # Extract Aadhaar number
    aadhaar_no = re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', english_text)
    if not aadhaar_no:
        aadhaar_no = re.search(r'\b\d{12}\b', english_text)
    aadhaar_no = aadhaar_no.group() if aadhaar_no else "Not found"
    
    # Extract Name (English)
    name_eng = re.search(r'(?:Name|NAME)[\s:]*([^\n]+)', english_text, re.IGNORECASE)
    if not name_eng:
        name_eng = re.search(r'\n([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\n', english_text)
    name_eng = name_eng.group(1).strip() if name_eng else "Not found"
    
    # Extract Name (Hindi)
    name_hin = extract_hindi_name(hindi_text)
    
    # Extract Date of Birth (improved pattern)
    dob = None
    dob_patterns = [
        r'(?:DOB|Date of Birth|Birth)[\s:]*(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(?:DOB|Date of Birth|Birth)[\s:]*(\d{2}-\d{2}-\d{4})',
        r'(?:DOB|Date of Birth|Birth)[\s:]*(\d{2}/\d{2}/\d{4})',
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    ]
    
    for pattern in dob_patterns:
        dob_match = re.search(pattern, english_text, re.IGNORECASE)
        if dob_match:
            dob = dob_match.group(1).strip()
            break
    
    dob = dob if dob else "Not found"
    
    # Extract Gender
    gender = re.search(r'(?:Gender|Sex)[\s:]*([^\n]+)', english_text, re.IGNORECASE)
    if not gender:
        gender = re.search(r'\b(Male|Female|M|F)\b', english_text, re.IGNORECASE)
    gender = gender.group(1).strip() if gender else "Not found"
    
    # Extract Address (English)
    address_lines = []
    address_pattern = r'(?:Address|Addr)[\s:]*([^\n]+(?:\n[^\n]+){0,3})'
    address_match = re.search(address_pattern, english_text, re.IGNORECASE)
    if address_match:
        address_lines = [line.strip() for line in address_match.group(1).split('\n') if line.strip()]
    address_eng = '\n'.join(address_lines) if address_lines else "Address not extracted"
    
    # Extract Address (Hindi)
    address_hin = extract_hindi_address(hindi_text, address_eng)
    
    # Process QR code data if available
    vid = "Not found"
    if qr_data:
        vid_match = re.search(r'vid="(\d+)"', qr_data)
        vid = vid_match.group(1) if vid_match else "Found in QR but no VID"
    
    return {
        'english': {
            'aadhaar_no': aadhaar_no,
            'name': name_eng,
            'dob': dob,
            'gender': gender,
            'address': address_eng,
            'vid': vid,
            'qr_data': qr_data or "No QR code detected"
        },
        'hindi': {
            'aadhaar_no': aadhaar_no,
            'name': name_hin,
            'dob': dob,
            'gender': 'पुरुष' if gender.lower() in ('male', 'm') else 'महिला',
            'address': address_hin,
            'vid': vid,
            'qr_data': qr_data or "QR कोड नहीं मिला"
        }
    }

def export_to_excel(data, filename='aadhaar_details.xlsx'):
    """Export extracted data to Excel file with Hindi support"""
    df = pd.DataFrame({
        'Field': ['Aadhaar Number', 'Name', 'Date of Birth', 'Gender', 'Address', 'Virtual ID', 'QR Data'],
        'English': [
            data['english']['aadhaar_no'],
            data['english']['name'],
            data['english']['dob'],
            data['english']['gender'],
            data['english']['address'],
            data['english']['vid'],
            data['english']['qr_data']
        ],
        'Hindi': [
            data['hindi']['aadhaar_no'],
            data['hindi']['name'],
            data['hindi']['dob'],
            data['hindi']['gender'],
            data['hindi']['address'],
            data['hindi']['vid'],
            data['hindi']['qr_data']
        ]
    })
    
    # Save with UTF-8 encoding by using ExcelWriter
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return filename

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file selected")
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('upload.html', error="No file selected")
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            eng_data, hin_data, eng_text, hin_text, qr_data = process_aadhaar(filepath)
            
            if eng_data is None or hin_data is None:
                return render_template('upload.html', error="Error processing Aadhaar card")
            
            aadhaar_details = extract_aadhaar_details(eng_data, hin_data, eng_text, hin_text, qr_data)
            excel_file = export_to_excel(aadhaar_details)
            
            return render_template('results.html', 
                               details=aadhaar_details,
                               excel_file=excel_file)
        else:
            return render_template('upload.html', error="Invalid file type. Only JPG, PNG allowed")
    
    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)