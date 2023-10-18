import time
import cv2
import numpy as np

def initialize_capture():
    cap = cv2.VideoCapture(0)
    return cap

def initialize_colors():
    red_color = (0, 0, 255)
    green_color = (0, 255, 0)
    return red_color, green_color

def detect_motion(previous_frame, current_frame):
    frame_difference = cv2.absdiff(previous_frame, current_frame)
    gray_difference = cv2.cvtColor(frame_difference, cv2.COLOR_BGR2GRAY)
    blurred_difference = cv2.GaussianBlur(gray_difference, (5, 5), 0)
    _, motion_mask = cv2.threshold(blurred_difference, 20, 255, cv2.THRESH_BINARY)
    dilated_motion = cv2.dilate(motion_mask, None, iterations=3)
    return dilated_motion


cap = initialize_capture()
red_color, green_color = initialize_colors()
zero_time = time.time()
red_duration = 5
green_duration = 10
previous_frame = cap.read()[1]
while cap.isOpened():
    zero_time = time.time()
    time_curr = 0
    while time_curr <= red_duration:
        time_curr = time.time() - zero_time
        current_frame = cap.read()[1]
        motion_detected = detect_motion(previous_frame, current_frame)
        contours, _ = cv2.findContours(motion_detected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_with_red_contours = np.zeros_like(current_frame)  # Черный фон
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue
            cv2.putText(previous_frame, "Status: {}".format("Motion detected"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, red_color, 3, cv2.LINE_AA)

            cv2.drawContours(black_with_red_contours, contours, -1, red_color, 2)
        cv2.putText(previous_frame, "Timer: {} sec".format(red_duration - time_curr.__int__()), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, green_color, 3, cv2.LINE_AA)
        cv2.imshow("Output", previous_frame)
        cv2.imshow("Motion Detection", black_with_red_contours)
        previous_frame = current_frame
        cv2.waitKey(1)
    while time_curr <= red_duration + green_duration:
        current_frame = cap.read()[1]
        cv2.putText(current_frame, "Status: {}".format("Motion Not Detected"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, green_color, 3, cv2.LINE_AA)
        cv2.putText(current_frame, "Timer: {} sec".format(red_duration + green_duration - time_curr.__int__()), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, green_color, 3, cv2.LINE_AA)
        cv2.imshow("Output", current_frame)
        time_curr = time.time() - zero_time
        cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
