import numpy as np
from util import base64_to_pil
from flask import Flask, request, jsonify, make_response, send_file

from flask_restful import Resource, Api
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from werkzeug.utils import secure_filename
from keras.utils import img_to_array

import os
import uuid

application = Flask(__name__)
api = Api(application)
CORS(application)

application.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root@localhost/healco'
application.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
application.config['SQLALCHEMY_RECORD_QUERIES'] = True
application.config['JWT_SECRET_KEY'] = 'apihealco'

db = SQLAlchemy(application)

model = tf.lite.Interpreter("data/model_10.tflite")
model.allocate_tensors()
input_details = model.get_input_details()
output_details = model.get_output_details()

# MODEL CORN DISEASE
class CornDisease(db.Model):
    __tablename__ = 'corn_disease'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(50), nullable=False)
    deskripsi = db.Column(db.TEXT)
    gejala = db.Column(db.TEXT)
    penyebab = db.Column(db.TEXT)
    pengobatan = db.Column(db.TEXT)
    gambar_1 = db.Column(db.String(250), nullable=False)
    gambar_2 = db.Column(db.String(250), nullable=False)
    gambar_3 = db.Column(db.String(250), nullable=False)

# CREATE DATABASE
application.app_context().push()
db.create_all()

class Home(Resource):
    def get(self):
        return make_response(jsonify({'status': 200, 'message': 'Welcome to HealCo API'}))

# NOTE : FUNGSI PREDICT MODEL
def model_predict(img, model):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    model.set_tensor(input_details[0]['index'], x)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    return results

# NOTE : PREDICT RESOURCE
class Predict(Resource):
    def post(self):
        application.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/uploads/predict/'

        target_names = ['bercak_daun', 'hawar_daun', 'karat_daun', 'daun_sehat']

        try:
            data = request.json
            img = base64_to_pil(data)

            uid = uuid.uuid4()
            filename = str(uid) + '.png'
            img.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

            pred = model_predict(img, model)

            hasil_label = target_names[np.argmax(pred)]
            hasil_prob = "{:.2f}".format(100 * np.max(pred))
            
            return make_response(jsonify({'status': '200', 'error': 'false', 'message': 'Berhasil melakukan prediksi', 'diagnosis': hasil_label, 'probability': hasil_prob, 'image': filename}))

        except Exception  as e:
            return make_response(jsonify({'status': '400', 'error': 'true', 'message': str(e), 'diagnosis': '', 'probability': '', 'email': '', 'image': ''}))

# NOTE : DETAIL CORN DISEASE
class Detail(Resource):
    def get(self, nama):
        try:
            disease = CornDisease.query.filter_by(nama=nama).first()
            if disease:
                return make_response(jsonify({'status': '200', 'error': 'false', 'nama': disease.nama, 'deskripsi': disease.deskripsi, 'gejala': disease.gejala, 'penyebab': disease.penyebab, 'pengobatan': disease.pengobatan, 'gambar': [disease.gambar_1, disease.gambar_2, disease.gambar_3]}))
            else:
                return make_response(jsonify({'status': '404', 'error': 'false', 'message': 'Penyakit tidak ditemukan'}))
        except Exception as e:
            return make_response(jsonify({'status': '400', 'error': 'true', 'message': str(e)}))

class ImagePredict(Resource):
    def get(self, filename):
        return send_file('uploads/predict/' + filename, mimetype='image/jpeg')

class ImageDisease(Resource):
    def get(self, filename):
        return send_file('uploads/disease/' + filename, mimetype='image/jpeg')

# NOTE : CORN DESEASE RESOURCE
class Disease(Resource):
    def post(self):
        application.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/uploads/disease'

        try:
            nama = request.form.get('nama')
            deskripsi = request.form.get('deskripsi')
            gejala = request.form.get('gejala')
            penyebab = request.form.get('penyebab')
            pengobatan = request.form.get('pengobatan')
            gambar_1 = request.files.get('gambar_1')
            gambar_2 = request.files.get('gambar_2')
            gambar_3 = request.files.get('gambar_3')

            uid = uuid.uuid4()

            filename1 = secure_filename(gambar_1.filename)
            image1 = str(uid) + filename1
            gambar_1.save(os.path.join(application.config['UPLOAD_FOLDER'], image1))

            filename2 = secure_filename(gambar_2.filename)
            image2 = str(uid) + filename2
            gambar_2.save(os.path.join(application.config['UPLOAD_FOLDER'], image2))

            filename3 = secure_filename(gambar_3.filename)
            image3 = str(uid) + filename3
            gambar_3.save(os.path.join(application.config['UPLOAD_FOLDER'], image3))

            cornDisease = CornDisease(nama=nama, deskripsi=deskripsi, gejala=gejala, penyebab=penyebab, pengobatan=pengobatan, gambar_1=image1, gambar_2=image2, gambar_3=image3)
            db.session.add(cornDisease)
            db.session.commit()
            return make_response(jsonify({'status': '200', 'error': 'false', 'message': 'Data Berhasil Ditambahkan'}))
        except Exception as e:
            return make_response(jsonify({'status': '400', 'error': 'true', 'message': str(e)}))

api.add_resource(Home, '/', methods=['GET'])
api.add_resource(Predict, '/predict', methods=['POST'])
api.add_resource(Detail, '/detail/<string:nama>', methods=['GET'])
api.add_resource(Disease, '/createdisease', methods=['POST'])
api.add_resource(ImagePredict, '/uploads/predict/<path:filename>', methods=['GET'])
api.add_resource(ImageDisease, '/uploads/disease/<path:filename>', methods=['GET'])

if __name__ == '__main__':
    application.run()