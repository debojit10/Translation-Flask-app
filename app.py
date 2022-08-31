import os
from flask import Flask, request, render_template, redirect
from flask import send_file, url_for, flash, url_for, send_from_directory
from werkzeug.utils import secure_filename
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import torch
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
model = model.to(device)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}
DOWNLOAD_FOLDER= 'downloads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['model'] = model
app.config['tokenizer'] = tokenizer
app.secret_key = 'dev'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def translate(filename):
    with open(filename) as f:
        text = f.read()
    texts = sent_tokenize(text)
    decoded = ''
    # You can also use "translate English to French" and "translate English to Romanian"
    for text in texts:
        inputs = tokenizer("translate English to German: "+text, return_tensors="pt").to(device)  # Batch size 1
        outputs = model.generate(input_ids=inputs["input_ids"],
                                max_length=512,
                                num_beams=5, 
                                no_repeat_ngram_size=2, 
                                early_stopping=True,)

        decoded += tokenizer.decode(outputs[0], skip_special_tokens=True)+'\n'

    output_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'output.txt')
    with open(output_path, "w") as text_file:
        text_file.write(decoded)

    return output_path

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('DownloadFile', name=filename))
    return '''
    <!doctype html>
    <title>Upload Text File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<name>')
def DownloadFile(name):
    path = translate(os.path.join(app.config['UPLOAD_FOLDER'], name))
    if not os.path.exists(path):
        return ("File {} not downloaded".format(path))
    try:
        # shutil.make_archive(path, "zip", app.config['UPLOAD_FOLDER'])
        return send_file(path, as_attachment=True)
    except Exception as e:
        pass