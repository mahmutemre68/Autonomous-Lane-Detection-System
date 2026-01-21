import cv2
import numpy as np


def make_coordinates(image, line_parameters):
    # producing coordinates from slope and intercption values (y = mx + b)
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0

    y1 = image.shape[0]  # bottom line of the video
    y2 = int(y1 * (1 / 2))

    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # 1. making up polynomes (y = mx + b) -> (slope, intercept)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]

        # slope filtering (remove horizontal lines)
        if abs(slope) < 0.3:
            continue

        # left lane(negative slope) vs right lane (positive slope)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    # taking mean
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    else:
        left_line = None

    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    else:
        right_line = None


    combined_lines = []
    if left_line is not None:
        combined_lines.append(left_line)
    if right_line is not None:
        combined_lines.append(right_line)

    return np.array(combined_lines)


def canny_edge_detector(image):
    # 1. convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2. reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 3. spot lines
    canny = cv2.Canny(blur, 50, 150)
    return canny


def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    bottom_left = (100, height)

    bottom_right = (width - 50, height)

    apex = (int(width / 2), int(height*0.5))
    # masking triangle
    triangle = np.array([
        [bottom_left, bottom_right, apex]
    ])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    # black screen
    line_image = np.zeros_like(image)

    # if lines spotted
    if lines is not None:
        for line in lines:
            # take the line's coordinates
            x1, y1, x2, y2 = line.reshape(4)
            # pull a red line (BGR: Blue=0, Green=0, Red=255)
            # boldness: 10 pixels
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)

    return line_image


# main program
cap = cv2.VideoCapture("test_video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # step 1: spot the lines
    canny_image = canny_edge_detector(frame)

    # step 2: focus on the road
    cropped_image = region_of_interest(canny_image)

    # step 3: HOUGH TRANSFORM (find the mathematicall lines)
    # rho=2: 2 pixels sensitivity
    # theta=np.pi/180: 1 degree sensitivity
    # threshold=100: En az 100 noktanın onayi lazım (kisa parazitleri eler)
    # minLineLength=40: 40 pikselden kisa çizgileri yoksay
    # maxLineGap=5: cizgiler arasında 5 piksel boşluk varsa onları birleştir
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100,
                            minLineLength=40, maxLineGap=5)

    # step 4: viusalize the lines in red
    averaged_lines = average_slope_intercept(frame, lines)

    line_image = display_lines(frame, averaged_lines)

    # step 5: original video and lines added on (Weighted Add)
    # Formula: Original * 0.8 + lines * 1 + 1 (Gamma)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    cv2.imshow("Lane Detection System", combo_image)

    # exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()










