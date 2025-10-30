from flask import Flask, render_template, request, jsonify
import easyocr
import cv2
import fitz  # PyMuPDF for handling PDFs
from openai import AzureOpenAI #importing
import os
import io
from PIL import Image
import tempfile
import numpy as np

app = Flask(__name__)

client = AzureOpenAI(
    api_key="5rYF0ToE5VWrlW5itIdq3RfmAkYB2Sr87i1nxXNmdGGDSQZnsSw4JQQJ99BHAC77bzfXJ3w3AAABACOGtqk2",  # Replace with your actual OpenAI API key
    api_version="2025-01-01-preview",
    azure_endpoint="https://icicipoc.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview"
)


# New route for handling camera captured images
@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    if 'camera_image' not in request.files:
        return jsonify({'error': 'No camera image received'}), 400

    file = request.files['camera_image']
    if file.filename == '':
        return jsonify({'error': 'No image data'}), 400

    try:
        # Read the image data
        image_data = file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL Image to numpy array for OpenCV
        opencv_image = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        if len(opencv_image.shape) == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        
        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, opencv_image)
            temp_file_path = temp_file.name
        
        try:
            # Extract text from the camera image
            extracted_text = extract_text_from_image(temp_file_path)
            
            # Get card details using OpenAI
            card_details = get_card_details(' '.join(extracted_text))
            
            return jsonify({'details': card_details})
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        return jsonify({'error': f'Error processing camera image: {str(e)}'}), 500




# Function to extract text using easyOCR
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])  # Specify 'en' for English language
    img = cv2.imread(image_path)
    result = reader.readtext(image_path, detail=0, width_ths=0.9)
    return result

# Function to extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Function to interact with Azure OpenAI to get card details
def get_card_details(extracted_text):
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[{"role": "system", "content": "Assistant is a large language model trained by OpenAI."},
                  {"role": "user", "content": f'''There will be four types of documents you may encounter:
                1. **Aadhaar Card**
                2. **PAN Card**
                3. **Driving License**
                4. **Passport**
                Please carefully follow the guidelines below based on the type of document. The content provided to you will be extracted text from the image or PDF of the document. Use the following instructions based on the type of document.
    
                    ---
    
                    ### If the extracted text is from an **Aadhaar Card**:
                    - **Extract the following fields**:
                    - **Name**: The full name of the individual. Look for text that appears to be a person's name, often appearing near the photo or in the main text area. Names may be in English or regional languages. Look for patterns like "Name:", or text that looks like a proper name (capitalized words that don't match other field labels).
                    - **Date of Birth**: The full date of birth (day, month, year). Look for "DOB:", "Date of Birth:", or date patterns like DD/MM/YYYY. If only the year is present, provide only the year. Do not infer or assume any details.
                    - **Sex (Gender)**: The gender of the individual (Male/Female/Other). Look for "Male", "Female", "M", "F", or regional language equivalents.
                    - **Address**: The address associated with the Aadhaar card. This is usually a longer text block with location details.
                    - **Aadhaar Number**: The 12-digit Aadhaar number, usually in format XXXX XXXX XXXX or as a continuous 12-digit number.
                
                    - **Instructions for missing or unclear data**:
                    - Carefully scan ALL extracted text for names - they might not be labeled explicitly as "Name:" 
                    - Look for any text that appears to be a person's name, even if it's not clearly labeled
                    - If any of the required fields (Name, Date of Birth, Sex, Address, Aadhaar Number) are not found in the text OR are unclear due to poor image quality, return **"Data not found"** for that specific field.
                    - If the image appears blurry or text is unclear, extract only what you can clearly read and mark unclear fields as "Data not found".
                    - If the **Date of Birth** only contains the year (e.g., "1990"), return only the year and **do not infer** the day or month.
                    - If the full Date of Birth is present with day, month, and year (e.g., "01 January 1990"), return the entire date as it is.
                    - Do not include any extra details, assumptions, or explanations. Just return the exact details in the required format.
    
                    **Output format**:
                    Provide the details in the following format (without asterisks or bold formatting):
                    Name: [Extracted Name or "Data not found"]
                    DOB: [Extracted Date of Birth or "Data not found"]
                    Gender: [Extracted Gender or "Data not found"]
                    Address: [Extracted Address or "Data not found"]
                    Aadhaar Number: [Extracted Aadhaar Number or "Data not found"]
                    
                    If multiple fields are unclear due to image quality, add this line at the end:
                    Note: Some fields are unclear due to poor image quality
                    
                    for example:
                        Name: Hariharan Saimurali
                        DOB: 27/02/2001
                        Gender: Male
                        Address: Mustur (VIF), Jagaluru (TQ), Davanagere (Dist), 577520
                        Aadhaar Number: 6420 6147 5245
                Do not include any unnecessary information or assumptions, and only provide the details as per the fields mentioned.
    
                ---
    
                ### If the extracted text is from a **PAN Card**:
                - **Extract the following fields**:
                - **Name**: The full name of the individual (as printed on the PAN card). Look carefully for any text that appears to be a person's name, even if not explicitly labeled.
                - **Father's Name**: The name of the father as printed on the PAN card.
                - **Date of Birth**: The full date of birth (day, month, year). If only the year is present, return only the year.
                - **PAN Number**: The 10-character Permanent Account Number (PAN) which typically follows a pattern: [A-Z]{5}[0-9]{4}[A-Z]{1}.

                - **Instructions for missing or unclear data**:
                - Carefully scan ALL extracted text for names - they might not be labeled explicitly
                - If any of the required fields (Name, Father's Name, Date of Birth, PAN Number) are not found in the text OR are unclear due to poor image quality, return **"Data not found"** for that field.
                - If the image appears blurry or text is unclear, extract only what you can clearly read and mark unclear fields as "Data not found".
                - If the **Date of Birth** only contains the year (e.g., "1990"), provide only the year, not the full date.
                - If the full Date of Birth is available (e.g., "01 January 1990"), provide the entire date.
    
                **Output format**:
                Provide the details in the following format (without asterisks or bold formatting):
                Name: [Extracted Name or "Data not found"]
                Father's Name: [Extracted Father's Name or "Data not found"]
                DOB: [Extracted Date of Birth or "Data not found"]
                PAN Number: [Extracted PAN Number or "Data not found"]
                
                If multiple fields are unclear due to image quality, add this line at the end:
                Note: Some fields are unclear due to poor image quality

                ---

                ### If the extracted text is from a **Driving License**:
                - **Extract the following fields**:
                - **Name**: The full name of the individual (as printed on the driving license). Look carefully for any text that appears to be a person's name, even if not explicitly labeled.
                - **Date of Birth**: The date of birth of the individual. Look for various labels like "DOB", "Date of Birth", "Born", "Birth Date", or dates that appear to be birth dates. Common formats include DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, MM/DD/YYYY, or written formats like "01 January 1990". If only the year is present, return only the year. Carefully examine all dates in the document to identify the birth date.
                - **Address**: The address associated with the driving license.
                - **License Number**: The driving license number (may also be labeled as "DL No", "License No", "Licence Number", etc.).
                - **Issue Date**: The date when the license was issued (may be labeled as "Issued", "Issue Date", "DOI", etc.).
                - **Expiry Date**: The date when the license expires (may be labeled as "Valid Till", "Expires", "Expiry", "Valid Upto", etc.).
                - **Vehicle Class**: The class of vehicles the license holder is authorized to drive (like LMV, MCWG, MCWOG, etc.).

                - **Instructions for missing or unclear data**:
                - Carefully scan ALL extracted text for names - they might not be labeled explicitly
                - If any of the required fields (Name, Date of Birth, Address, License Number, Issue Date, Expiry Date, Vehicle Class) are not found in the text OR are unclear due to poor image quality, return **"Data not found"** for that field.
                - If the image appears blurry or text is unclear, extract only what you can clearly read and mark unclear fields as "Data not found".
                - For **Date of Birth**: Pay special attention to find birth date - it might not be explicitly labeled as "DOB". Look for patterns like dates that would make sense as birth dates (typically older dates compared to issue/expiry dates).
                - If the **Date of Birth** only contains the year (e.g., "1990"), provide only the year, not the full date.
                - If the full Date of Birth is available (e.g., "01 January 1990"), provide the entire date.

                **Output format**:
                Provide the details in the following format (without asterisks or bold formatting):
                Name: [Extracted Name or "Data not found"]
                DOB: [Extracted Date of Birth or "Data not found"]
                Address: [Extracted Address or "Data not found"]
                License Number: [Extracted License Number or "Data not found"]
                Issue Date: [Extracted Issue Date or "Data not found"]
                Expiry Date: [Extracted Expiry Date or "Data not found"]
                Vehicle Class: [Extracted Vehicle Class or "Data not found"]
                
                If multiple fields are unclear due to image quality, add this line at the end:
                Note: Some fields are unclear due to poor image quality

                ---

                ### If the extracted text is from a **Passport**:
                - **Extract the following fields**:
                - **Name**: The full name of the individual (as printed on the passport). Look carefully for any text that appears to be a person's name, even if not explicitly labeled.
                - **Date of Birth**: The full date of birth (day, month, year). If only the year is present, return only the year.
                - **Place of Birth**: The place where the individual was born.
                - **Passport Number**: The passport number.
                - **Issue Date**: The date when the passport was issued.
                - **Expiry Date**: The date when the passport expires.
                - **Nationality**: The nationality of the passport holder.
                - **Gender**: The gender of the individual (Male/Female/Other).

                - **Instructions for missing or unclear data**:
                - Carefully scan ALL extracted text for names - they might not be labeled explicitly
                - If any of the required fields (Name, Date of Birth, Place of Birth, Passport Number, Issue Date, Expiry Date, Nationality, Gender) are not found in the text OR are unclear due to poor image quality, return **"Data not found"** for that field.
                - If the image appears blurry or text is unclear, extract only what you can clearly read and mark unclear fields as "Data not found".
                - If the **Date of Birth** only contains the year (e.g., "1990"), provide only the year, not the full date.
                - If the full Date of Birth is available (e.g., "01 January 1990"), provide the entire date.

                **Output format**:
                Provide the details in the following format (without asterisks or bold formatting):
                Name: [Extracted Name or "Data not found"]
                DOB: [Extracted Date of Birth or "Data not found"]
                Place of Birth: [Extracted Place of Birth or "Data not found"]
                Passport Number: [Extracted Passport Number or "Data not found"]
                Issue Date: [Extracted Issue Date or "Data not found"]
                Expiry Date: [Extracted Expiry Date or "Data not found"]
                Nationality: [Extracted Nationality or "Data not found"]
                Gender: [Extracted Gender or "Data not found"]
                
                If multiple fields are unclear due to image quality, add this line at the end:
                Note: Some fields are unclear due to poor image quality

            PLEASE DONT INCLUDE ANY KIND OF NOTE OR EXPLANATION OR INTRODUCTION , JUST DETAILS IS ENOUGH. DO NOT USE ASTERISKS OR BOLD FORMATTING IN THE OUTPUT. IF THE IMAGE IS BLURRY OR TEXT IS UNCLEAR, ONLY EXTRACT WHAT YOU CAN CLEARLY READ AND MARK THE REST AS "Data not found". CAREFULLY LOOK FOR NAMES IN THE TEXT EVEN IF NOT EXPLICITLY LABELED AS "Name:". SCAN ALL TEXT FOR PERSON NAMES.
            HERE IS THE EXTRACTED TEXT :{extracted_text}
    '''}])
    return response.choices[0].message.content


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure the 'uploads' directory exists
    upload_folder = 'static/uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Save the uploaded file to the 'uploads' directory
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Process file based on its extension
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(file_path)
    elif file_extension in ['png', 'jpg', 'jpeg']:
        # Extract text from the image
        extracted_text = extract_text_from_image(file_path)
    else:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    card_details = get_card_details(' '.join(extracted_text))

    return jsonify({'details': card_details})

if __name__ == '__main__':
    app.run(debug=True)

