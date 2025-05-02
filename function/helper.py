import math
from function.utils_rotate import correct_common_misreads, is_valid_plate


# Tính phương trình đường thẳng đi qua hai điểm
def linear_equation(x1, y1, x2, y2):
    b = y1 - (y2 - y1) * x1 / (x2 - x1)
    a = (y1 - b) / x1
    return a, b

# Kiểm tra một điểm có nằm gần đường thẳng nối hai điểm không
def check_point_linear(x, y, x1, y1, x2, y2):
    a, b = linear_equation(x1, y1, x2, y2)
    y_pred = a * x + b
    return math.isclose(y_pred, y, abs_tol=1)  # Sử dụng sai số nhỏ hơn

# Phân tích kết quả YOLOv8 để ghép chuỗi biển số
def read_plate(yolo_license_plate, im):
    LP_type = "1"

    # Dự đoán với YOLOv8 (sử dụng `.predict()`)
    results = yolo_license_plate(im)

    if isinstance(results, list):
        bb_list = results[0].boxes.data.tolist()
    elif hasattr(results, 'pandas'):
        bb_list = results.pandas().xyxy[0].values.tolist()
    else:
        print("Không có kết quả hợp lệ từ YOLO.")
        return "unknown"

    # Kiểm tra số lượng bounding boxes hợp lệ
    if len(bb_list) < 2 or len(bb_list) > 20:

        print("Không đủ số lượng bounding boxes để nhận diện biển số.")
        return "unknown"

    center_list = []
    y_sum = 0
    for bb in bb_list:
        # Tính toán tọa độ trung tâm của mỗi bounding box
        x_c = (bb[0] + bb[2]) / 2
        y_c = (bb[1] + bb[3]) / 2
        y_sum += y_c
        class_id = int(bb[-1])
        char = yolo_license_plate.names[class_id]  # Chuyển ID thành ký tự
        center_list.append([x_c, y_c, char])

    print("Center list:", center_list)

    # Xác định trái - phải để xét tuyến tính
    l_point = min(center_list, key=lambda p: p[0])
    r_point = max(center_list, key=lambda p: p[0])

    for ct in center_list:
        if l_point[0] != r_point[0] and not check_point_linear(
                ct[0], ct[1], l_point[0], l_point[1], r_point[0], r_point[1]
        ):
            LP_type = "2"
            break

    # Tính toán trung bình tọa độ Y
    y_mean = int(y_sum / len(bb_list))
    line_1, line_2 = [], []

    if LP_type == "2":
        # Nếu LP_type là 2, chia bounding boxes thành 2 dòng
        for c in center_list:
            if int(c[1]) > y_mean:
                line_2.append(c)
            else:
                line_1.append(c)
        # Sắp xếp và kết hợp lại các ký tự từ 2 dòng
        sorted_chars = sorted(line_1, key=lambda x: x[0]) + [["-", "-", "-"]] + sorted(line_2, key=lambda x: x[0])
    else:
        # Nếu LP_type là 1, sắp xếp các ký tự theo tọa độ X
        sorted_chars = sorted(center_list, key=lambda x: x[0])

    # Tạo biển số từ các ký tự đã sắp xếp
    license_plate = "".join(str(c[2]) for c in sorted_chars if c[2] != "-")

    print("Biển số sau khi ghép:", license_plate)

    if LP_type == "2":
        # Nếu có 2 dòng, ghép biển số theo định dạng
        license_plate = (
                "".join(str(c[2]) for c in sorted(line_1, key=lambda x: x[0])) +
                "-" +
                "".join(str(c[2]) for c in sorted(line_2, key=lambda x: x[0]))
        )

    # Chỉnh sửa các lỗi phổ biến trong việc đọc biển số
    license_plate = correct_common_misreads(license_plate)

    # Kiểm tra xem biển số có hợp lệ không
    if not is_valid_plate(license_plate):
        return "unknown"

    return license_plate
