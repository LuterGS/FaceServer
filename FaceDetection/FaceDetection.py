import face_recognition
import numpy as np
from PIL import Image
import FaceDetection.__regression as regression
import FaceDetection.__library as library

# FaceDetection 클래스에서 턱선을 구하기 위한 회귀신경망 미리 세팅
do_regression = regression.all_regression(3, 0.05)

nose_point = [15.094, 16.667, 17.605, 18.349, 19.227, 20.645]  # 작을수록 코가 작다는 뜻
mouse_point = [30.54, 32.646, 33.552, 34.774, 36.626, 38.796]  # 작을수록 입이 작다는 뜻
chin_point = [0.014, 0.031, 0.063, 0.172, 0.237, 0.314]  # 작을수록 턱이 날카롭다는 뜻
face_garo_point = [0.794, 0.808, 0.817, 0.824, 0.833, 0.849]  # 작을수록 볼이 작다는 뜻


def get_range_of_factor(nose, mouse, chin, face_garo):
    """
    코, 입, 턱선, 볼살 길이를 작다~크다 분류로 7개로 나누기 위한 함수.
    나누기 위한 포인트는 nose_point, mouse_point, chin_point, face_garo_point에 사전 정의된 값을 사용한다.
    """

    data = [nose, mouse, chin, face_garo]
    data_range = [nose_point, mouse_point, chin_point, face_garo_point]
    ranges = [0, 0, 0, 0]

    for i in range(4):
        if data_range[i][0] > data[i]:
            ranges[i] = 1
        elif data_range[i][0] <= data[i] < data_range[i][1]:
            ranges[i] = 2
        elif data_range[i][1] <= data[i] < data_range[i][2]:
            ranges[i] = 3
        elif data_range[i][2] <= data[i] < data_range[i][3]:
            ranges[i] = 4
        elif data_range[i][3] <= data[i] < data_range[i][4]:
            ranges[i] = 5
        elif data_range[i][4] <= data[i] < data_range[i][5]:
            ranges[i] = 6
        elif data_range[i][5] <= data[i]:
            ranges[i] = 7
    return ranges


class FaceDetection:

    def __init__(self):
        pass

    def get_pic(self, image_path):
        __image = face_recognition.load_image_file(image_path)
        __face_landmarks_list = face_recognition.face_landmarks(__image, model="large")

        # face_recognition으로 추출한 특징을 개별분리한다.
        self.chin = __face_landmarks_list[0]['chin']
        self.left_eyebrow = __face_landmarks_list[0]['left_eyebrow']
        self.right_eyebrow = __face_landmarks_list[0]['right_eyebrow']
        self.nose_bridge = __face_landmarks_list[0]['nose_bridge']
        self.nose_tip = __face_landmarks_list[0]['nose_tip']
        self.left_eye = __face_landmarks_list[0]['left_eye']
        self.right_eye = __face_landmarks_list[0]['right_eye']
        self.top_lip = __face_landmarks_list[0]['top_lip']
        self.bottom_lip = __face_landmarks_list[0]['bottom_lip']
        self.x_size, self.y_size = FaceDetection.get_image_xy(__image)

    def get_points(self):
        garo_gradient = library.get_gradient(self.chin[0], self.chin[-1])
        garo_length = library.get_2d_length(self.chin[0], self.chin[-1])
        self.garo_line = library.일차함수_detailed(library.일차함수(self.x_size, self.y_size, self.chin[0], garo_gradient))

        sero_gradient = library.get_gradient(self.nose_bridge[0], self.nose_bridge[-1])
        self.sero_line = library.일차함수_detailed(
            library.일차함수(self.x_size, self.y_size, library.get_middle(self.nose_bridge[0], self.nose_bridge[-1]),
                         sero_gradient))
        sero_length, sero_chin_pos, _ = library.get_horizontal_length(self.sero_line, self.chin, self.nose_bridge[0])

        # 오른쪽 눈의 가로 퍼센트
        factor_1 = library.get_percentage(garo_length,
                                          library.get_2d_length(library.get_가까운점(self.garo_line, self.right_eye[0]),
                                                                library.get_가까운점(self.garo_line, self.right_eye[3])))

        # 오른쪽 눈의 세로 퍼센트
        eye_top, eye_bottom = library.top_bottommost(self.right_eye)
        factor_2 = library.get_percentage(sero_length, library.get_2d_length(library.get_가까운점(self.sero_line, eye_top),
                                                                             library.get_가까운점(self.sero_line,
                                                                                              eye_bottom)))

        # 코의 가로 퍼센트
        factor_3 = library.get_percentage(garo_length,
                                          library.get_2d_length(library.get_가까운점(self.garo_line, self.nose_tip[0]),
                                                                library.get_가까운점(self.garo_line, self.nose_tip[-1])))

        # 입의 가로 퍼센트
        lip_left, lip_right = library.left_rightmost(self.bottom_lip)
        factor_4 = library.get_percentage(garo_length, library.get_2d_length(library.get_가까운점(self.garo_line, lip_left),
                                                                             library.get_가까운점(self.garo_line,
                                                                                              lip_right)))

        # 오른쪽 눈매의 기울기
        factor_5 = library.get_angle_by_gradient(library.get_gradient(self.right_eye[3], self.right_eye[0]),
                                                 garo_gradient)

        # 얼굴 가로길이 : 가로길이-코 뺀 길이 = 1 : N
        nosetip_left, nosetip_right = library.left_rightmost(self.nose_tip)
        nosetip_length = library.get_2d_length(library.get_가까운점(self.garo_line, nosetip_left),
                                               library.get_가까운점(self.garo_line, nosetip_right))
        factor_6 = (garo_length - nosetip_length) / garo_length

        # 턱각도 (3차함수의 기울기 정도)
        factor_7 = self.get_chin_a()
        factor_7 = float(factor_7)

        # 눈과 눈 사이의 길이
        factor_8 = self.get_between_eye_length(garo_gradient, garo_length)

        # 가로길이대 세로길이의 비율
        factor_9 = 1 + round(sero_length / garo_length, 3)

        result = [[factor_1, factor_2, factor_3, factor_4, factor_5, factor_6, factor_7, factor_8, factor_9]]
        point_result = get_range_of_factor(factor_3, factor_4, factor_7, factor_6)
        result = FaceDetection.refine_data(result)
        result = np.asarray(result)

        return result, point_result

    def get_between_eye_length(self, garo_gradient, garo_length):
        left_end = self.left_eye[3]
        right_end = self.right_eye[0]
        lr_gradient = library.get_gradient(left_end, right_end)
        if garo_gradient == lr_gradient:
            lr_length = library.get_2d_length(left_end, right_end)
        else:
            lr_length = library.get_2d_length(
                library.get_가까운점(self.garo_line, left_end),
                library.get_가까운점(self.garo_line, right_end))
        lr_percentage = library.get_percentage(garo_length, lr_length)
        return lr_percentage

    def get_chin_a(self):

        chin_regression_data = library.get_chin_input(self.chin)
        do_regression.input_data(chin_regression_data[0], chin_regression_data[1])
        do_regression.training(600)

        return library.get_float_from_variable(do_regression.variables[0])

    @staticmethod
    def get_image_xy(image_info):
        draw = Image.fromarray(image_info)
        return draw.size[0], draw.size[1]

    @staticmethod
    def refine_data(data):
        # 데이터는 (데이터 - 원래 데이터의 최대값) / (원래 데이터의 최대값 - 원래 데이터의 최솟값)
        # 으로 정제해 0~1 사이의 값을 가지게끔 다듬음
        data[0][0] = (data[0][0] - 12.625) / 8.046
        data[0][1] = data[0][1] / 12.725  # 0빼는거라 생략
        data[0][2] = (data[0][2] - 7.268) / 20.946
        data[0][3] = (data[0][3] - 23.273) / 22.551
        data[0][4] = (data[0][4] + 0.274) / 0.332
        data[0][5] = (data[0][5] - 0.718) / 0.209
        data[0][6] = (data[0][6] - 0.001) / 0.534
        data[0][7] = (data[0][7] - 22.37) / 10.418
        data[0][8] = (data[0][8] - 1.65) / 0.285

        # 사전에 정의된 값으로 데이터를 정규화해주되, 0보다 작거나 1보다 클 경우 0~1로 최적화시켜줌.
        for i in range(len(data[0])):
            if data[0][i] > 1:
                data[0][i] = 1
            if data[0][i] < 0:
                data[0][i] = 0
        return data
