from flask import Flask, render_template, request, send_file, jsonify
import csv
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from models import dataset_ops, vectorization_ops

app = Flask(__name__)

data_file = "data.csv"
if not os.path.exists(data_file):
    with open(data_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Good", "Bad", "Datetime"])

model = tf.keras.models.load_model('full_convolution_combined.keras')

data_train, _, _ = dataset_ops.load_data(split_ratio=0.3, random_state=42)
word_vectorizer = vectorization_ops.create_word_vectorizer(data_train['url'])
char_vectorizer = vectorization_ops.create_char_vectorizer(data_train['url'])

best_threshold = np.load("fpr_tpr/best_threshold_fccombi.npy")

def check_url_with_model(url):

    url_word_vector = word_vectorizer.transform([url]).toarray()
    url_char_vector = char_vectorizer.texts_to_sequences([url])

    input_length = 200
    if url_word_vector.shape[1] < input_length:
        word_padding = np.zeros((1, input_length - url_word_vector.shape[1]))
        url_word_vector_padded = np.hstack((url_word_vector, word_padding))
    else:
        url_word_vector_padded = url_word_vector[:, :input_length]
    
    url_char_vector_padded = tf.keras.preprocessing.sequence.pad_sequences(
        url_char_vector, maxlen=input_length, padding='post'
    )

    probability = model.predict(
        (url_word_vector_padded, url_char_vector_padded)
    )[0][0]

    if probability >= best_threshold:
        return "ĐỘC HẠI", probability
    else:
        return "AN TOÀN", probability

@app.route('/add_evaluate', methods=['GET', 'POST'])
def add_evaluate():
    if request.method == 'POST':
        url = request.form.get('url')
        status = request.form.get('status')
        if url and status:

            rows = []
            if os.path.exists(data_file):
                with open(data_file, mode="r") as file:
                    reader = csv.reader(file)
                    next(reader)  
                    rows = [row for row in reader]

            found = False
            for row in rows:
                if row[0] == url:
                    found = True
                    if status == "Good":
                        row[1] = str(int(row[1]) + 1)
                    elif status == "Bad":
                        row[2] = str(int(row[2]) + 1)
                    row[3] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break

            if not found:
                good_count = 1 if status == "Good" else 0
                bad_count = 1 if status == "Bad" else 0
                rows.append([url, str(good_count), str(bad_count), datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            with open(data_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["URL", "Good", "Bad", "Datetime"])
                writer.writerows(rows)

    rows = []
    if os.path.exists(data_file):
        with open(data_file, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  
            rows = [row for row in reader]

    return render_template('add_evaluate.html', rows=rows)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            message, probability = check_url_with_model(url)
            response = {
                "url": url,
                "status": message,
                "probability": round(probability, 2)
            }
            if request.headers.get('Content-Type') == 'application/json':
                return jsonify(response)
            else:
                return render_template('home.html', message=f"URL '{url}' là {message} với xác suất {response['probability']:.2f}")

    return render_template('home.html', message=None)

@app.route('/download')
def download_file():

    rows = []
    if os.path.exists(data_file):
        with open(data_file, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  
            rows = [row for row in reader]

    download_rows = []
    for row in rows:
        url = row[0]
        good = int(row[1])
        bad = int(row[2])
        status = "Good" if good >= bad else "Bad"

        download_rows.append([url, status])

    temp_file = "filtered_data.csv"
    with open(temp_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Status"]) 
        writer.writerows(download_rows)  

    return send_file(temp_file, as_attachment=True)

@app.route('/api')
def api_documentation():
    return render_template('api.html')

if __name__ == '__main__':
    app.run(debug=True)
