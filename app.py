# import numpy as np
# from util import base64_to_pil
# from flask import Flask, request, jsonify, make_response
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image


# import tensorflow as tf

# application = Flask(__name__)

# # model = load_model('data/model_3.h5')

# model = tf.lite.Interpreter("data/model_6.tflite")
# model.allocate_tensors()
# input_details = model.get_input_details()
# output_details = model.get_output_details()

# print('=============================')
# print(input_details)
# print(output_details)
# print('=============================')



# # NOTE : FUNGSI PREDICT MODEL
# def model_predict(img, model):
#     img = img.resize((224, 224))
#     x = image.img_to_array(img)
#     x = x.reshape(-1, 224, 224, 3)
#     x = x.astype('float32')
#     x = x / 255.0
#     # pred = model.predict(x)
    
#     model.set_tensor(input_details[0]['index'], x)
#     model.invoke()
#     output_data = model.get_tensor(output_details[0]['index'])
#     results = np.squeeze(output_data)
#     return results


# @application.route('/', methods=['GET'])
# def index():
#     return make_response(jsonify({'status': 200, 'message': 'Ini API Healco'}))

# @application.route('/predict', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         img = base64_to_pil(request.json)


#         preds = model_predict(img, model)

#         target_names = ['bintik_daun', 'hawar_daun', 'karat_daun', 'daun_sehat']
#         hasil_label = target_names[np.argmax(preds)]
#         hasil_prob = "{:.2f}".format(100 * np.max(preds)) 

#         return make_response(jsonify({'status': 200, 'diagnosis': hasil_label, 'probability': hasil_prob}))
#     return None

# if __name__ == '__main__':
#     application.run(debug=True)