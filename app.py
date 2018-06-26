from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config['UPLOADS'] = os.path.join('static/uploads/')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f_list = request.files.getlist('filename[]')
        for idx, f in enumerate(f_list):
            fn = 'uploadfile{}.jpeg'.format(idx)
            full_path = os.path.join(app.config['UPLOADS'], fn)
            f.save(full_path)

        result = get_classification(full_path)

        return render_template('show_results.html', im_url=full_path, classification=result)
    if request.method == 'GET':
        return render_template('upload.html')

def get_classification(file_path):
    return "yes"

if __name__ == '__main__':
    print(Flask)