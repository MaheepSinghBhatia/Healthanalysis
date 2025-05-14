from flask import Flask, render_template, request, redirect, url_for
import os
from docx import Document
import fitz  # PyMuPDF
import nltk
from nltk.data import find
from dotenv import load_dotenv
import textwrap
import numpy as np
import requests
import json
import time
import re

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK tokenizer is available
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load environment variables from the key.env file
load_dotenv("key.env")
SECRET_API_KEY = os.getenv("API_KEY")
REFERER = os.getenv("REFERER")
TITLE = os.getenv("TITLE")

def query(prompt, system_prompt="You are a helpful medical assistant.", retries=3, delay=3):
    headers = {
        "Authorization": f"Bearer {SECRET_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": REFERER,
        "X-Title": TITLE
    }
    payload = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }

    attempt = 0
    wait_time = delay

    while attempt < retries:
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=30  # <-- Important: Add timeout
            )
            response.raise_for_status()
            result = response.json()

            if 'choices' not in result or not result['choices']:
                return "❌ Error: Invalid API response format."

            return result['choices'][0]['message']['content'].strip()

        except requests.exceptions.RequestException as e:
            print(f"⚠️ API request error: {e}")
            # Handle rate limiting with retry
            if isinstance(e, requests.exceptions.HTTPError) and response.status_code == 429:
                time.sleep(wait_time)
                wait_time *= 2
            else:
                break  # Don't retry for other errors
        attempt += 1

    return "❌ Error: Failed after multiple retries or network error."

def read_txt(file):
    return file.read().decode('utf-8')

def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return ''.join([page.get_text() for page in doc])

def read_docx(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def split_text(text, max_length=1000):
    return textwrap.wrap(text, width=max_length, break_long_words=False, break_on_hyphens=False)

def summarize_health_doc(text, max_chunks=3):
    chunks = split_text(text)
    chunks = chunks[:max_chunks]
    summaries = []

    for chunk in chunks:
        s = query(f"Summarize this medical text in plain language for a patient/user to make it easier for them to understand:\n\n{chunk}")
        if s and not s.startswith("❌"):
            summaries.append(s)

    if not summaries:
        return "❌ Error: Unable to generate a valid summary."
    combined = " ".join(summaries)
    final_summary = query(f"Summarize the following clearly for patients in one paragraph:\n\n{combined}")
    return final_summary.strip() if final_summary.strip() else "❌ Error: Summary generation failed."

def generate_qa(text):
    prompt = f"""Based on the following content, generate 5 relevant, patient-friendly questions and clear answers.

Format as:
- **Question?** Answer.

Content:
{text[:3000]}
"""
    result = query(prompt)
    return result.strip() if result else "❌ Error: Failed to generate Q&A."

def extract_links(markdown_text):
    return re.findall(r'\[.*?\]\((https?://[^\s)]+)\)', markdown_text)

def get_references_and_links(text):
    prompt = f"""Please provide 3-5 credible resources for users related to the document provided to learn more, including links in markdown format. Classify/describe each link.

Content:
{text[:3000]}
"""
    raw_links = query(prompt)
    return raw_links if raw_links else "❌ Error: Failed to generate links."

def classify_medical_content(text):
    medical_keywords = ['diagnosis', 'treatment', 'disease', 'symptom', 'hospital', 'medicine', 'surgery', 'doctor', 'patient', 'clinical']
    medical_score = sum(text.lower().count(word) for word in medical_keywords)

    if medical_score < 3:
        return "Non-Medical"
    
    categories = ['Cardiology', 'Neurology', 'Oncology', 'Orthopedics', 'General Medicine']
    return np.random.choice(categories)

def process_uploaded_file(file):
    if file.filename.endswith(".pdf"):
        text = read_pdf(file)
    elif file.filename.endswith(".docx"):
        text = read_docx(file)
    elif file.filename.endswith(".txt"):
        text = read_txt(file)
    else:
        return None, "Unsupported file type."

    if not text.strip():
        return None, "Document is empty or unreadable."

    category = classify_medical_content(text)
    summary = summarize_health_doc(text)
    qa = generate_qa(text)
    links = get_references_and_links(text)

    result = f"📋 Summary:\n\n{summary}\n\n❓ Q&A:\n\n{qa}\n\n🌐 Resources:\n\n{links}"
    return result, category

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return redirect(url_for('home'))

    try:
        result, category = process_uploaded_file(file)
        if result:
            return render_template('result.html', result=result, category=category)
        else:
            return render_template('error.html', error_message="Error: Unsupported file type or empty document."), 400
    except Exception as e:
        print("⚠️ Internal Error:", e)
        return render_template('error.html', error_message="Internal server error during processing."), 500

if __name__ == '__main__':
    app.run(debug=True)
