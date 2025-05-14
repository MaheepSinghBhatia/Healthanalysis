# server.py

from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
from io import BytesIO
import app as analyzer  # Import your Streamlit logic module

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For flash messaging

# Global store for last prediction (simple way)
latest_result = BytesIO()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global latest_result
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        result_text, category = analyzer.process_uploaded_file(file)
        if result_text is None:
            flash('Failed to process the file: ' + category)
            return redirect(url_for('index'))
        
        # Save to in-memory buffer for download
        latest_result = BytesIO()
        latest_result.write(result_text.encode('utf-8'))
        latest_result.seek(0)

        flash(f'✅ File processed successfully! Category: {category}')
        flash(f'You can now download your results below.')
        return redirect(url_for('index'))

@app.route('/download')
def download():
    global latest_result
    if latest_result.getbuffer().nbytes == 0:
        flash('No file processed yet!')
        return redirect(url_for('index'))
    return send_file(latest_result, as_attachment=True, download_name="health_doc_summary.txt")

if __name__ == '__main__':
    app.run(debug=True)
