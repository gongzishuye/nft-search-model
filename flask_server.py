import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

from faiss_app import search_nft

UPLOAD_FOLDER = '/home/ec2-user/workspace/facenet/upload_file'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


collection_ctr_kv = {}
with open('database_nft_address.txt') as f:
    for line in f:
        tokens = line.strip().split(' ')
        collection_ctr_kv[tokens[-1]] = tokens[0]


def show_num(num):
    num = num.split('.')[0]
    for idx, c in enumerate(num):
        if c != '0':
            break
    return int(num[idx:])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/search', methods=['GET', 'POST'])
def search():
    print('request received.')
    # check if the post request has the file part
    if 'file' not in request.files:
        return {
                'code': -1,
                'data': {}
            }
    file = request.files['file']
    print(file.filename)
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return {
                'code': 1,
                'data': {}
            }
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        des_nft = search_nft(file_path)
        if des_nft is None:
            return {
                'code': 2,
                'data': {}
            }
        items = des_nft.split('/')
        ctr = collection_ctr_kv[items[-2]]
        idx = show_num(items[-1])
        return {
                'code': 0,
                'data': {
                    "contract": ctr,
                    "idx": idx,
                }
            }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8099)
