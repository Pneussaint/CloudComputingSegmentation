from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
from io import BytesIO
from segmentation.model import segmentation, get_weights
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('upload.html')

@app.route('/segImage', methods=['POST'])
def segImage():
    # Get the image file from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file sent'})
    image_file = request.files['image']

    # Open the image using PIL
    image = Image.open(image_file)
    
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
    app.run(port=80, host='0.0.0.0', debug=True)