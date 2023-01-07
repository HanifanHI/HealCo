import numpy as np
from util import base64_to_pil
from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.preprocessing import image


app = Flask(__name__)

model = load_model('models/model_2.h5')

def model_predict(img, model):
    img = img.resize((224, 224))
    
    x = image.img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)

        #==================================================================================#

        target_names = ['bintik_daun', 'hawar_daun', 'karat_daun', 'daun_sehat']     # ⚠️ SESUAIKAN ⚠️

        hasil_label = target_names[np.argmax(preds)]
        hasil_prob = "{:.2f}".format(100 * np.max(preds)) # 2f adalah presisi angka dibelakang koma (coba ganti jadi 0f, 3f, dst)

        #==================================================================================#

        return jsonify(result=hasil_label, probability=hasil_prob)

    return None

if __name__ == '__main__':
    # OPTION 1: NORMAL SERVE THE APP
    app.run(debug=True)
    #app.run(port=5002, threaded=False)

    # OPTION 2: SERVE THE APP WITH GEVENT
    # Setiap merubah file di main.js, ubah juga 5000, menjadi 5001, dst (alasan: cache)
    #http_server = WSGIServer(('0.0.0.0', 5050), app)
    #http_server.serve_forever()