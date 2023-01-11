import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from discord.ext import commands
import uuid
import requests
import shutil

from faiss_app import search_nft



''' bot
'''
# Credentials
TOKEN = 'OTcyMzI3Mjc1MDcwNjMxOTY2.YnXcVw.YyNR2OV4jAEwU7FSaYra_X-Vv7Q'

# Create bot
client = commands.Bot(command_prefix='!')

# Startup Information
@client.event
async def on_ready():
    print('Connected to bot: {}'.format(client.user.name))
    print('Bot ID: {}'.format(client.user.id))


# Command
@client.command()
async def helloworld(ctx):
    await ctx.send('Hello World!')


@client.command()
async def go(ctx):
    print('message from discord.')
    # USAGE: use command .save in the comment box when uploading an image to save the image as a jpg
    try:
        url = ctx.message.attachments[0].url            # check for an image, call exception if none found
    except IndexError:
        print("Error: No attachments")
        await ctx.send("No attachments detected!")
    else:
        if url[0:26] == "https://cdn.discordapp.com":   # look to see if url is from discord
            r = requests.get(url, stream=True)
            image_name = str(uuid.uuid4()) + '.jpg'      # uuid creates random unique id to use for image names
            image_name = 'upload/' + image_name
            with open(image_name, 'wb') as out_file:
                print('Saving image: ' + image_name)
                shutil.copyfileobj(r.raw, out_file)     # save image (goes to project directory)
            des_nft = search_nft(image_name)
            if des_nft is None:
                await ctx.send('Not Found')
                return
            items = des_nft.split('/')
            ctr = collection_ctr_kv[items[-2]]
            idx = show_num(items[-1])
            url = URL_TEMPLATE.format(ctr, idx)
            # html = opensea_html.format(url)
            await ctx.send(url)
        else:
            print(f'{url} is not normal!')
''' bot
'''


UPLOAD_FOLDER = '/home/ec2-user/workspace/facenet/upload_file'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
URL_TEMPLATE = 'https://opensea.io/assets/{}/{}'


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


opensea_html = '''
    <!doctype html>
    <a href="{}">Go To OpenSea</a>
'''

notfound_html = '''
    <!doctype html>
    <h1>Not Found</h1>
'''

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
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            des_nft = search_nft(file_path)
            if des_nft is None:
                return notfound_html
            items = des_nft.split('/')
            ctr = collection_ctr_kv[items[-2]]
            idx = show_num(items[-1])
            url = URL_TEMPLATE.format(ctr, idx)
            html = opensea_html.format(url)
            return html
    return '''
    <!doctype html>
    <title>Upload NFT Picture</title>
    <h1>Upload NFT Picture</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Search>
    </form>
    '''


if __name__ == '__main__':
    client.run(TOKEN)
