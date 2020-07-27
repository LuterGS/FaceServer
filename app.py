from flask import Flask, render_template, request, Response
from flask_cors import CORS, cross_origin
import os
from werkzeug.utils import secure_filename
from PIL import UnidentifiedImageError
import FaceDetection.__Training_NN as nn_20
import FaceDetection.FaceDetection as FaceDetection
import FaceDetection.__library as library

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

neuralnetwork_20 = nn_20.FaceNN(9, 20)
neuralnetwork_eye = nn_20.FaceNN(2, 7)
face_detection = FaceDetection.FaceDetection()


def get_pic_result(file_path):

    try:
        face_detection.get_pic(file_path)
    except IndexError:
        print("얼굴인식이 실패함")
        return Response('{"message":"face detection failed"}', status=400, mimetype='application/json')
    except UnidentifiedImageError:
        print("이미지 확장자명 실패")
        return Response('{"message":"not an image file"}', status=400, mimetype='application/json')
    except:
        print("기타 에러")
        return Response('Server Internal Error', status=500, mimetype='application/json')
    data, range_data = face_detection.get_points()
    eye_data = [[data[0][0], data[0][1]]]

    result_facetype = neuralnetwork_20.predict_data(data, '/home/lutergs/FaceServer/FaceDetection/Data/weights')
    result_eyetype = neuralnetwork_eye.predict_data(eye_data, '/home/lutergs/FaceServer/FaceDetection/Data/weights_eye')

    return Response(library.get_json(result_facetype, result_eyetype, range_data), status=200, mimetype='application/json')

"""
@app.route('/')
def index():
    return render_template('index.html')
"""

@app.route('/face', methods=['GET', 'POST'])
def get_pic():
    if request.method == 'POST':
        # print(request.files)
        file = request.files['file']
        file_name = secure_filename(file.filename)
        # print(app.instance_path)
        file_path =  os.path.join('/home/lutergs/FaceServer', 'Pic', file_name)
        file.save(file_path)
        print(file_path)

        result = get_pic_result(file_path)
        print(request.files, result)

        desl = 'rm -rf ' + file_path
        os.system(desl)
        return result
    if request.method == 'GET':
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
