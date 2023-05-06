from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    target_dir = 'uploads/'
    file.save(target_dir + file.filename)

    return 'The file {} has been uploaded and saved.'.format(file.filename)

if __name__ == '__main__':
    app.run(debug=True)
