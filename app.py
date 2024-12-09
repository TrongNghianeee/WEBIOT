from flask import Flask, render_template, request, send_file
import csv
import os
from datetime import datetime

app = Flask(__name__)

data_file = "data.csv"
if not os.path.exists(data_file):
    with open(data_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Good", "Bad", "Datetime"])

@app.route('/add_evaluate', methods=['GET', 'POST'])
def add_evaluate():
    if request.method == 'POST':
        url = request.form.get('url')
        status = request.form.get('status')
        if url and status:
            # Read current data
            rows = []
            if os.path.exists(data_file):
                with open(data_file, mode="r") as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    rows = [row for row in reader]

            # Check if URL already exists
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

            # If URL is new, add it
            if not found:
                good_count = 1 if status == "Good" else 0
                bad_count = 1 if status == "Bad" else 0
                rows.append([url, str(good_count), str(bad_count), datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

            # Write updated data back to file
            with open(data_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["URL", "Good", "Bad", "Datetime"])
                writer.writerows(rows)

    # Read data to display
    rows = []
    if os.path.exists(data_file):
        with open(data_file, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            rows = [row for row in reader]

    return render_template('add_evaluate.html', rows=rows)

@app.route('/', methods=['GET', 'POST'])
def home():
    message = None
    if request.method == 'POST':
        url = request.form.get('url')
        if url:
            # Read current data
            rows = []
            if os.path.exists(data_file):
                with open(data_file, mode="r") as file:
                    reader = csv.reader(file)
                    next(reader)  # Skip header
                    rows = [row for row in reader]

            # Check if URL exists and evaluate
            for row in rows:
                if row[0] == url:
                    good = int(row[1])
                    bad = int(row[2])
                    message = "Good" if good > bad else "Bad"
                    break
            else:
                message = "URL not found in the database."

    return render_template('home.html', message=message)

@app.route('/download')
def download_file():
    return send_file(data_file, as_attachment=True)

@app.route('/clear', methods=['POST'])
def clear_data():
    # Clear all data in the CSV file
    with open(data_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["URL", "Good", "Bad", "Datetime"])
    return render_template('add_evaluate.html', rows=[])

if __name__ == '__main__':
    app.run(debug=True)