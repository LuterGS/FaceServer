from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import FaceDetection.__Training_NN as nn_20
import FaceDetection.FaceDetection as FaceDetection

app = Flask(__name__)

neuralnetwork_20 = nn_20.FaceNN(9, 20)
neuralnetwork_eye = nn_20.FaceNN(2, 7)
face_detection = FaceDetection.FaceDetection()


def get_pic_result(file_path):
    face_detection.get_pic(file_path)
    data, range_data = face_detection.get_points()
    eye_data = [[data[0][0], data[0][1]]]

    result_facetype = neuralnetwork_20.predict_data(data, '/home/lutergs/Development/FaceServer/FaceDetection/Data/weights')
    result_eyetype = neuralnetwork_eye.predict_data(eye_data, '/home/lutergs/Development/FaceServer/FaceDetection/Data/weights_eye')

    return result_facetype, result_eyetype, range_data


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_pic', methods=['GET', 'POST'])
def get_pic():
    if request.method == 'POST':
        # print(request.files)
        file = request.files['file']
        file_name = secure_filename(file.filename)
        # print(app.instance_path)
        file_path =  os.path.join('/home/lutergs/Development/FaceServer', 'Pic', file_name)
        file.save(file_path)

        result = get_pic_result(file_path)
        print(request.files, result)

        return str(result)


if __name__ == '__main__':
    app.run(debug=True)
