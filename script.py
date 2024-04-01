import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np

from pprint import pprint
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from torch.utils.data import DataLoader
from flask import Flask, request, jsonify, send_file
from PIL import Image
from io import BytesIO
import requests
import torchvision.transforms as T
from segmentation.model import segmentation, get_weights
app = Flask(__name__)

@app.route('/segImage', methods=['POST'])
def segImage():
    # Get the image file from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file sent'})
    image_file = request.files['image']

    # Open the image using PIL
    image = Image.open(image_file)
    w = image.width
    h = image.height
    if(not( h % 32 == 0 and w % 32 == 0)):
        image = image.resize((h + 32 - h % 32, w + 32 - w % 32))
    
    # Perform the segmentation
    img = segmentation(image)
    img = Image.fromarray(img)
    # Save the image to a BytesIO object
    img_io = BytesIO()
    img.save(img_io, 'JPEG', )
    img_io.seek(0)

    # Return the image as a response
    return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name='segmented.jpg')


if __name__ == "__main__":
    app.run()