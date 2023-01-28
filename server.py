import numpy as np
from util import base64_to_pil
from flask import Flask, request, jsonify, make_response
from keras.models import load_model

from flask_restful import Resource, Api
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from werkzeug.security import generate_password_hash, check_password_hash
from keras.utils import img_to_array

# from keras.preprocessing import image
# from werkzeug.utils import secure_filename

import os
import uuid
from datetime import datetime, timedelta

app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root@localhost/healcodb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['JWT_SECRET_KEY'] = 'apihealco'

jwt = JWTManager(app)
db = SQLAlchemy(app)

model = load_model('data/model_3.h5')

# MODEL ========================================================
# MODEL USER
class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(250), nullable=False)
    email = db.Column(db.String(100), nullable=False, unique=True)
    no_hp = db.Column(db.String(12))
    password = db.Column(db.String(250), nullable=False)
    profile = db.Column(db.String(250), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow() + timedelta(hours=7))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow() + timedelta(hours=7))


# MODEL PREDICT HISTORY
class PredictHistory(db.Model):
    __tablename__ = 'predict_history'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(250), nullable=False)
    accuracy = db.Column(db.String(10), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    image = db.Column(db.String(250), nullable=False)


# MODEL CORN DISEASE
class CornDisease(db.Model):
    __tablename__ = 'corn_disease'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama = db.Column(db.String(50), nullable=False)
    deskripsi = db.Column(db.TEXT)
    gejala = db.Column(db.TEXT)
    pengobatan = db.Column(db.TEXT)
    gambar_1 = db.Column(db.String(250), nullable=False)
    gambar_2 = db.Column(db.String(250), nullable=False)
    gambar_3 = db.Column(db.String(250), nullable=False)


# MODEL TANGGAPAN
class Tanggapan(db.Model):
    __tablename__ = 'tanggapan'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    email = db.Column(db.String(100), nullable=False)
    tanggapan = db.Column(db.TEXT)

# CREATE DATABASE ====================================================================
app.app_context().push()
db.create_all()

# NOTE : REGISTER RESOURCE
class Register(Resource):
    def post(self):
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            no_hp = request.form.get('no_hp')
            password = request.form.get('password')

            profile = 'default_profile.png'
            gen_password = generate_password_hash(password)

            user = User(name=name, email=email, no_hp=no_hp, password=gen_password, profile=profile)
            db.session.add(user)
            db.session.commit()
            return make_response(jsonify({'status': 200, 'message': 'Sukses menambahkan data user'}))

        except:
            return make_response(jsonify({'status': 400, 'message': 'Gagal menambahkan data user'}))


# NOTE : LOGIN RESOURCE
class Login(Resource):
    def post(self):
        try:
            email = request.form.get('email')
            password = request.form.get('password')

            user = User.query.filter_by(email=email).first()

            if not user:
                return make_response(jsonify({'message': 'Email tidak terdaftar'}))
            if not check_password_hash(user.password, password):
                return make_response(jsonify({'message': 'Password salah'}))

            offset = datetime.utcnow() + timedelta(hours=7)
            expires = offset + timedelta(seconds=30)
            token = create_access_token({'email': email, 'exp': expires})
            return make_response(jsonify({'status': '200', 'message': 'Berhasil Login', 'email': email, 'token': token}))

        except Exception as e:
            return make_response(jsonify({'status': 400, 'message': str(e)}))


# NOTE : FUNGSI PREDICT MODEL
def model_predict(img, model):
    img = img.resize((224, 224))
    x = img_to_array(img)
    x = x.reshape(-1, 224, 224, 3)
    x = x.astype('float32')
    x = x / 255.0
    pred = model.predict(x)
    return pred


# NOTE : PREDICT RESOURCE
class Predict(Resource):
    @jwt_required()
    def post(self):
        app.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/uploads/predict/'
        target_names = ['bintik_daun', 'hawar_daun', 'karat_daun', 'daun_sehat']

        current_user = get_jwt_identity()
        email = current_user['email']

        try:
            data = request.json
            img = base64_to_pil(data)

            uid = uuid.uuid4()
            filename = str(uid) + '.png'
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            pred = model_predict(img, model)

            hasil_label = target_names[np.argmax(pred)]
            hasil_prob = "{:.2f}".format(100 * np.max(pred))
            
            history = PredictHistory(name=hasil_label, accuracy=str(hasil_prob), email=email, image=filename)
            db.session.add(history)
            db.session.commit()
            return make_response(jsonify({'status': 200, 'diagnosis': hasil_label, 'probability': hasil_prob, 'email': email, 'image': filename}))

        except Exception as e:
            return make_response(jsonify({'status': 400, 'message': str(e)}))

# NOTE : DETAIL CORN DISEASE
class Detail(Resource):
    @jwt_required()
    def get(self, nama):
        try:
            disease = CornDisease.query.filter_by(nama=nama).first()
            if disease:
                return make_response(jsonify({'nama': disease.nama, 'deskripsi': disease.deskripsi, 'gejala': disease.gejala, 'pengobatan': disease.pengobatan, 'gambar': [disease.gambar_1, disease.gambar_2, disease.gambar_3]}))
            else:
                return make_response(jsonify({'message': 'Penyakit tidak ditemukan'}), 404)
        except Exception as e:
            return make_response(jsonify({'Error': str(e)}), 500)


# NOTE : PREDICT HISTORY RESOURCE
class History(Resource):
    @jwt_required()
    def get(self, email):
        try:
            query = PredictHistory.query.filter_by(email=email)

            if query:
                output = [{
                'id': data.id,
                'name': data.name,
                'accuracy': data.accuracy,
                'image': data.image } for data in query]
                return make_response(jsonify(output), 200)
            else:
                return make_response(jsonify({'message': 'Email tidak ditemukan'}))
        except Exception as e:
            return make_response(jsonify({'Error': str(e)}))

# NOTE : DELETE HISTORY RESOURCE
class DeleteHistory(Resource):
    @jwt_required()
    def delete(self, id):
        try:
            query = PredictHistory.query.get(id)

            db.session.delete(query)
            db.session.commit()
            return make_response(jsonify({'message': 'Data berhasil dihapus'}))
        except Exception as e:
            return make_response(jsonify({'Error': str(e)}))


# NOTE : TANGGAPAN RESOURCE
class TanggapanResource(Resource):
    @jwt_required()
    def post(self):
        try:
            tanggapan= request.form.get('tanggapan')

            current_user = get_jwt_identity()
            email = current_user['email']

            if tanggapan:
                dataModel = Tanggapan(email=email, tanggapan=tanggapan)
                db.session.add(dataModel)
                db.session.commit()
                return make_response(jsonify({'message': 'Tanggapan anda telah kami terima'}))
            return make_response(jsonify({'message': 'Data tidak boleh kosong'}))
        except Exception as e:
            return make_response(jsonify({'Error': str(e), 'status': 500}))


# NOTE : EDIT PROFILE
class EditProfile(Resource):
    @jwt_required()
    def get(self, email):
        try:
            user = User.query.filter_by(email=email).first()

            output = {
                'id': user.id,
                'name': user.name,
                'email': user.email,
                'no_hp': user.no_hp,
                'profile': user.profile
            }
            return make_response(jsonify(output), 200)
        except Exception as e:
            make_response(jsonify({'message': 'Gagal get data'}))

    @jwt_required()
    def put(self, email):
        app.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/uploads/profile/'

        try:
            user = User.query.filter_by(email=email).first()
            
            profile = request.files.get('profile')

            uid = uuid.uuid4()
            filename = str(uid) + '.png'
            profile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            user.name = request.form.get('name')
            user.no_hp = request.form.get('no_hp')
            user.profile = filename
            user.updated_at = datetime.utcnow() + timedelta(hours=7)

            db.session.commit()
            return make_response(jsonify({'status': 200, 'message': 'Berhasil edit data'}))
        except Exception as e:
            make_response(jsonify({'Error ' + str(e)}))


# NOTE : CORN DESEASE RESOURCE
# class Disease(Resource):
#     def post(self):
#         app.config['UPLOAD_FOLDER'] = os.path.realpath('.') + '/uploads/disease'

#         try:
#             nama = request.form.get('nama')
#             deskripsi = request.form.get('deskripsi')
#             gejala = request.form.get('gejala')
#             pengobatan = request.form.get('pengobatan')
#             gambar_1 = request.files.get('gambar_1')
#             gambar_2 = request.files.get('gambar_2')
#             gambar_3 = request.files.get('gambar_3')

#             uid = uuid.uuid4()
            

#             filename1 = secure_filename(gambar_1.filename)
#             image1 = str(uid) + filename1
#             gambar_1.save(os.path.join(app.config['UPLOAD_FOLDER'], image1))

#             filename2 = secure_filename(gambar_2.filename)
#             image2 = str(uid) + filename2
#             gambar_2.save(os.path.join(app.config['UPLOAD_FOLDER'], image2))

#             filename3 = secure_filename(gambar_3.filename)
#             image3 = str(uid) + filename3
#             gambar_3.save(os.path.join(app.config['UPLOAD_FOLDER'], image3))

#             cornDisease = CornDisease(nama=nama, deskripsi=deskripsi, gejala=gejala, pengobatan=pengobatan, gambar_1=image1, gambar_2=image2, gambar_3=image3)
#             db.session.add(cornDisease)
#             db.session.commit()
#             return make_response(jsonify({'status': 200, 'message': 'Data Berhasil Ditambahkan'}))
#         except Exception as e:
#             return make_response(jsonify({'status': 401, 'message': str(e)}))

api.add_resource(Register, '/register', methods=['POST'])
api.add_resource(Login, '/login', methods=['POST'])
api.add_resource(Predict, '/predict', methods=['POST'])
api.add_resource(Detail, '/detail/<string:nama>', methods=['GET'])
api.add_resource(History, '/history/<string:email>', methods=['GET'])
api.add_resource(DeleteHistory, '/history/<id>', methods=['DELETE'])
api.add_resource(TanggapanResource, '/tanggapan', methods=['POST'])
api.add_resource(EditProfile, '/editprofile/<string:email>', methods=['GET', 'PUT'])
# api.add_resource(Disease, '/createdisease', methods=['POST'])


if __name__ == '__main__':
    app.run()