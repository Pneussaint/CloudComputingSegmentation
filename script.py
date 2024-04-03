import csv
from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
from io import BytesIO, StringIO
from segmentation.model import segmentation, get_prediction_score
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

@app.route('/getWeights', methods=['POST'])
def get_weights():
    # Get the image file from the request
    if 'image' not in request.files:
        return jsonify({'error': 'No file sent'})
    image_file = request.files['image']

    # Open the image using PIL
    image = Image.open(image_file)
    pred_score, pred_class = get_prediction_score(image)
    donnees = [{'Classe': pred_class[i], 'Score': pred_score[i]} for i in range(len(pred_class))]
     # Création d'un objet StringIO pour écrire dans le fichier CSV
    stringIO = StringIO()
    writer = csv.DictWriter(stringIO, fieldnames=['Classe', 'Score'])

    # Écriture des données dans le fichier CSV
    writer.writeheader()
    for row in donnees:
        writer.writerow(row)

    output = BytesIO()
    output.write(stringIO.getvalue().encode('utf-8'))
    # Retourne le contenu du fichier CSV au client
    output.seek(0)

    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='scores.csv')


if __name__ == "__main__":
    app.run(port=80, host='0.0.0.0', debug=True)