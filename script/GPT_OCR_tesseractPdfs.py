"""More sophisticated version of a image2text converter using fitz for pdf2image, tesseract for OCR. 
    Uses full pdfs. Transcribe engine control. 
"""


import openai
import fitz  # PyMuPDF
import sys, pymupdf  # alternative to fitz
import pytesseract
from pytesseract import Output
from PIL import Image
import io
from dotenv import load_dotenv 
import os
from openai import OpenAI
from tqdm import tqdm

print("script running…")

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to extract images from PDF
def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        for img in page.get_images(full=True): # seems to only take one pic per page
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                image = Image.open(io.BytesIO(image_bytes)) # OLD
                # image = Image.open(io.BytesIO(image_bytes)).convert("RGB") # NEW
                images.append(image)
            except Exception as e:
                print(f"Could not identify image file: {e}")
    return images

# Function to apply OCR to images and extract Swedish text
def apply_ocr_to_images(images):
    ocr_results = []
    for image in images:
        text = pytesseract.image_to_string(image, lang='eng', output_type=Output.STRING) # does swe work here?
        ocr_results.append(text)
    return ocr_results

# Function to transcribe text using OpenAI API
def transcribe_texts(ocr_texts):
    transcriptions = []
    for ocr_text in ocr_texts:
        response = client.chat.completions.create(model="gpt-4o", 
            messages=[
            {"role": "user",
                # "content": f"What’s in this image?\n\n{ocr_text}",
                "content": f"This is a english OCR text using tesseract, the text was originally Swedish however and hence the text is incorrect in many places. Correct the text to proper Swedish. \n\n{ocr_text}",}],
            max_tokens=4096) # adjust this for output sizes gpt-4o = 4096
        content = response.choices[0].message.content
        transcriptions.append(content)
    return transcriptions

#dynamically load pdfs
def load_pdfs(directory_path):
    pdf_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            # Check if the file has a .pdf extension
            if file.lower().endswith('.pdf'):
                # Get the full path of the pdf file and add it to the list
                full_path = os.path.join(root, file)
                pdf_files.append(full_path)

    return pdf_files

# Main function
def main(pdf_files):
    # Pipeline for the pdfs
    with tqdm(total=len(pdf_files), desc="Processing PDFs") as pbar:
        for pdf_path in pdf_files:
            images = extract_images_from_pdf(pdf_path)
            ocr_texts = apply_ocr_to_images(images)
            # print(f"Here comes the OCR: {ocr_texts}") # only for trouble shooting
            transcriptions = transcribe_texts(ocr_texts)
            
            # Save the transcription to a text file named after the PDF file
            output_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}.txt"
            with open(output_filename, 'w', encoding='utf-8') as f:
                for idx, transcription in enumerate(transcriptions):
                    f.write(f"Transcription for image {idx + 1}:\n{transcription}\n\n")
            
            print(f"Transcriptions saved to {output_filename}")
            pbar.update(1)



if __name__ == "__main__":    
    directory_path = '/Users/kailashdejesushornig/Documents/GitHub/Stipendier/data/pdfs/testSubset'  # Replace with your directory path
    pdf_files = load_pdfs(directory_path)

    main(pdf_files)

    """Comments on S_Åke_Bäckman: 10 seconds processing on 2 pages
    - Göteborg den 4 september 2022,--> should ave been 4 september 1969, hallucination... 
    - känner --> känna 
    - name, title, address and signatures have not been retained. 
    
    on GMF_Dahlbergfonden: 
    - Error in process: PIL.UnidentifiedImageError:  cannot identify image file, typically means that the data being read is not a valid image. This can happen for a few reasons, such as if the data being extracted from the PDF is not actually an image, or if there’s an issue with how the image data is being processed.
    
    on Adlerbertska_Chalmersfonden: 
    - poor quality in OCR page 1 = ['jot\n\noot\nco\n\nwe\nhe\n\n', ''] --> infeasible corrections

    General comments: the model sometimes chooses to add comments to it's response, helpful but the human must skip them of course.

    """

