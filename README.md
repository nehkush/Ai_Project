# Ai_Project
import google.generativeai as genai
import gradio as gr
import pytesseract
import pdfplumber
from PIL import Image
import os

# Set API key directly in the code
API_KEY = "AIzaSyCXJNDtFBcRTlxq3pmTneJ-Q4Qz-evbwdA"

# Configure Gemini API with the provided key
genai.configure(api_key=API_KEY)

# Hardcoded prompt to compare two texts
HARD_CODED_PROMPT = """
Compare the following two texts and provide a similarity score from 0 to 100:

Reference Text: {reference_text}
Student Answer: {student_text}

Give only the score, nothing else.
"""

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text.strip() if text else "No text found in PDF."
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

# Function to extract text from an image using OCR
def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image).strip() if image else "No text found in image."
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

# Function to get similarity score using Gemini API
def get_similarity_score_gemini(reference_text, student_text):
    try:
        # Truncate texts if they exceed the model's length limit
        reference_text = reference_text[:4000]  # Limit text length for model
        student_text = student_text[:4000]      # Limit text length for model
        
        # Prepare the prompt for the Gemini API
        prompt = HARD_CODED_PROMPT.format(reference_text=reference_text, student_text=student_text)
        
        # Call Gemini API to get the similarity score
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        
        # Return similarity score if the response is valid
        return float(response.text.strip()) if response.text.strip().replace('.', '', 1).isdigit() else "Invalid response from API."
    except Exception as e:
        return f"Error processing similarity: {str(e)}"

# Gradio Interface to upload files and display similarity result
def check_similarity(pdf_file, student_file):
    try:
        # Extract text from the reference PDF
        reference_text = extract_text_from_pdf(pdf_file.name)
        
        # Check for valid reference text
        if "Error" in reference_text:
            return reference_text
        
        # Extract text from the student file (image or PDF)
        if student_file.name.endswith(".jpg") or student_file.name.endswith(".png"):
            student_text = extract_text_from_image(student_file.name)
        else:
            student_text = extract_text_from_pdf(student_file.name)
        
        # Check for valid student text
        if "Error" in student_text:
            return student_text
        
        # Get similarity score synchronously
        score = get_similarity_score_gemini(reference_text, student_text)
        
        return f"Similarity Score: {score}%"
    except Exception as e:
        return f"Error in processing similarity: {str(e)}"

# Gradio interface
def launch_demo():
    with gr.Blocks() as demo:
        gr.Markdown("### AI Copy Checker: Compare Student Answers with Reference PDF")
        ref_pdf = gr.File(label="Upload Reference PDF")
        student_ans = gr.File(label="Upload Student Answer (Image or PDF)")
        result_output = gr.Textbox()
        submit_btn = gr.Button("Check Similarity")
        submit_btn.click(check_similarity, inputs=[ref_pdf, student_ans], outputs=result_output)

    demo.launch(debug=True)

# Run the Gradio interface
if __name__ == "__main__":
    launch_demo()
