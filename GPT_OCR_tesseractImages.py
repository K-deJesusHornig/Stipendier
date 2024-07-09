""" The pytesseract.image_to_string seems to only cope with english letter, the rest of the work is correct though. 
--> An LLM can correct for this.  

input: PNG, JPEG and TIFF.

Remaining problems: 
1. The text output from this pipeline is croped after a while, why? 
2. there are more languages like Swedish, but somehow they seems hard to access. 
tesseract 5.4.1 --> only shows 3 languages, while there should be 100+
3. Tesseact can be run directly from CLI with flags:  "tesseract data/images/test.png output_file -l eng"  

For full documentation on tesseract
https://github.com/tesseract-ocr/tesseract/blob/main/README.md
https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc 
"""

import pytesseract # Python-tesseract is a wrapper for Google’s Tesseract-OCR Engine.
# print(pytesseract.__version__) #  0.3.10  Released: Aug 16, 2022
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print("Script running...")

# Load the local image
image_path = '/Users/kailashdejesushornig/Documents/GitHub/Stipendier/data/images/Screenshot 2024-07-09 at 10.00.42.png'
image = Image.open(image_path)

# Apply OCR to the image to extract text
# Available languages: print(pytesseract.get_languages(config='')) 
# --> currently compatible with ['eng', 'osd'= Orientation and Script Detection, 'snum'= Sparse Text Number]
ocr_text = pytesseract.image_to_string(image, lang="eng") #https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html 



# Use the extracted text as input to OpenAI API
response = client.chat.completions.create(model="gpt-4o",
messages=[
    {
        "role": "user",
        # "content": f"What’s in this image?\n\n{ocr_text}",
        "content": f"This is a english OCR using tesseract, the text was Swedish however and hence the text is incorrect in many places. Correct the text to proper Swedish. \n\n{ocr_text}",
    }
],
max_tokens=500) # adjust this for output sizes

# Verbose/headless 
# print(f"######This is the OCR_TEXT:###### \n{ocr_text}")
# print(f"\n######Corrected version#####\n{content}") 

#  Content to be written to file
content = response.choices[0].message.content
absolute_path = '/Users/kailashdejesushornig/Documents/GitHub/Stipendier/corrected_output.txt'

# Write the content to the file
with open(absolute_path, 'w', encoding='utf-8') as file:
    file.write(content)

print(f"Content successfully written to {absolute_path}")