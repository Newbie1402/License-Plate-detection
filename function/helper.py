import math
from .utils_rotate import correct_common_misreads, is_valid_plate


# Tính phương trình đường thẳng đi qua hai điểm
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

# Kiểm tra một điểm có nằm gần đường thẳng nối hai điểm không
def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=3)  # Sai số 3 px

# Phân tích kết quả YOLO để ghép chuỗi biển số
def read_plate(yolo_license_plate, im):
    LP_type = "1"
    results = yolo_license_plate(im)
    bb_list = results.pandas().xyxy[0].values.tolist()

    if len(bb_list) == 0 or len(bb_list) < 7 or len(bb_list) > 10:
        return "unknown"

    center_list = []
    y_sum = 0
    for bb in bb_list:
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        center_list.append([x_c, y_c, bb[-1]])

    # Xác định trái - phải để xét tuyến tính
    l_point = min(center_list, key=lambda p: p[0])
    r_point = max(center_list, key=lambda p: p[0])

    for ct in center_list:
        if l_point[0] != r_point[0] and not check_point_linear(
            ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]
        ):
            LP_type = "2"
            break

    y_mean = int(y_sum / len(bb_list))
    line_1, line_2 = [], []

    if LP_type == "2":
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        sorted_chars = sorted(line_1, key=lambda x: x[0]) + [["-", "-", "-"]] + sorted(line_2, key=lambda x: x[0])
    else:
        sorted_chars = sorted(center_list, key=lambda x: x[0])

    license_plate = "".join(str(c[2]) for c in sorted_chars if c[2] != "-")
    if LP_type == "2":
        license_plate = (
            "".join(str(c[2]) for c in sorted(line_1, key=lambda x: x[0])) +
            "-" +
            "".join(str(c[2]) for c in sorted(line_2, key=lambda x: x[0]))
        )

    license_plate = correct_common_misreads(license_plate)

    if not is_valid_plate(license_plate):
        return "unknown"

    return license_plate