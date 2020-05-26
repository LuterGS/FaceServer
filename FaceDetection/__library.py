import math
import numpy as np


def get_middle(ar1, ar2):
    return (ar1[0] + ar2[0]) / 2, (ar1[1] + ar2[1]) / 2


def get_2d_length(ar1, ar2):
    output = math.sqrt(
        math.pow(ar1[0] - ar2[0], 2) +
        math.pow(ar1[1] - ar2[1], 2)
    )

    return round(output, 3)


def get_gradient(ar1, ar2):
    try:
        output = (ar2[1] - ar1[1]) / (ar2[0] - ar1[0])
    except:
        output = "90도"
    return output


def left_rightmost(face_landmark):
    # 눈길이 양측 두 좌표 구하기
    left = [10000, 0]
    right = [0, 0]
    for i in face_landmark:
        if i[0] < left[0]:
            left = i
        if i[0] > right[0]:
            right = i
    return left, right


def top_bottommost(face_landmark):
    top = [0, 10000]
    bottom = [0, 0]
    for i in face_landmark:
        if i[1] < top[1]:
            top = i
        if i[0] > bottom[1]:
            bottom = i
    return top, bottom


def 일차함수(x, y, pos, grad):
    """
    :param x: 이미지의 x좌표 사이즈
    :param y: 이미지의 y좌표 사이즈
    :param pos: 한 점의 좌표
    :param grad: 함수의 기울기
   :return:
    """
    if grad == "90도":
        test = []
        for i in range(y):
            test.append((pos[0], i))
        return test

    b = pos[1] - (grad * pos[0])
    x_pos = np.arange(0, x, 1)  # -10부터 10까지 0.1 간격의 수.
    y_pos = [(grad * num + b) for num in x_pos]
    value = []
    for i in range(len(x_pos)):
        if 0 < x_pos[i] < x and y_pos[i] > 0 and y_pos[i] < y:
            value.append((x_pos[i], int(y_pos[i])))
    return value


def 일차함수_detailed(pos):
    y_length = pos[-1][1] - pos[0][1]
    val = []
    if abs(y_length) > len(pos) and y_length > 0:
        for i in range(len(pos) - 1):
            for j in range(pos[i + 1][1] - pos[i][1]):
                val.append((pos[i][0], pos[i][1] + j))
    elif abs(y_length) > len(pos) and y_length < 0:
        for i in range(len(pos) - 1):
            for j in range(pos[i][1] - pos[i + 1][1]):
                val.append((pos[i][0], pos[i][1] - j))
    else:
        val = pos
    return val


def get_가까운점(line, pos):
    distance = 1000
    for i in range(len(line)):
        distance2 = get_2d_length(line[i], pos)
        if distance2 < distance:
            distance = distance2
            close_pos = line[i]
    return close_pos


def get_highest(y):
    output = -1000

    for i in range(len(y)):
        if y[i] > output:
            output = y[i]

    return output


def get_chin_input(data):
    test = []
    output = []
    for i in range(len(data)):
        test.append(float(data[i][0]))
        output.append([test[i], data[i][1]])

    output = np.array(output).transpose()
    output_0 = np.mean(output[0])
    output[0] -= output_0
    output[0] /= 2

    output_1_highest = get_highest(output[1])
    output[1] -= output_1_highest
    output[1] *= -1

    output[0] /= 2
    output[1] *= 2

    for i in range(len(output[1])):
        if output[0][i] < 0:
            output[1][i] *= -1

    return output


def get_horizontal_length(horizontal_line, chin, nosebridge_top):
    distance = 1000
    for i in range(len(chin)):
        for j in range(len(horizontal_line)):
            distance2 = get_2d_length(chin[i], horizontal_line[j])
            if distance2 < distance:
                distance = distance2
                linepos = horizontal_line[j]

    get_ylength = get_2d_length(linepos, nosebridge_top)
    return get_ylength, linepos, nosebridge_top


def get_float_from_variable(variable):
    output = str(variable)
    output = output[56:-1]

    return output


def get_percentage(big, small):
    output = small / big
    return round(output * 100, 3)


def get_angle_by_gradient(grad1, grad2):
    x = (grad1 - grad2) / (1 + grad1 * grad2)
    return round(x, 3)
