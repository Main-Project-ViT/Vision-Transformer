from flask import Flask, render_template, request
from predict import input_reshape
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
    file = request.files['image']
    target_dir = 'uploads/'
    filename = (target_dir + file.filename)
    file.save(filename)
    res = input_reshape(filename)
    os.remove(filename)
    return render_template('index.html', response=res)

# @app.route('/results', methods=['POST'])
# def results():
    

if __name__ == '__main__':
    app.run(debug=True)
