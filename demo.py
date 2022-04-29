import cv2 as cv
import numpy as np
import height_estimation_module as hem
from datetime import date
from pathlib import Path

IMAGE_PATH = str(Path.cwd().joinpath("images/lums.png"))
print(IMAGE_PATH)
WINDOW_NAME = "image"

drawing = False
mode = True
ix, iy = -1, -1
scale_factor = 173 / 313
point_list = []


def draw_distance(event, x, y, flags, param):
    global point_list, img

    if event == cv.EVENT_LBUTTONDOWN:
        point_list.append(np.array([x, y]))
        cv.circle(img, (x, y), 2, (0, 255, 0), -1)

        if len(point_list) == 2:
            a, b = point_list
            cv.line(img, a, b, (0, 255, 0), 2)
            distance = np.sqrt(np.sum((a - b) ** 2)) * scale_factor
            distance = hem.estimate_height(distance, rhs)
            midpoint = ((point_list[0] + point_list[1]) / 2).astype(int)
            cv.putText(
                img,
                f"{distance:.2f}m",
                midpoint,
                cv.FONT_HERSHEY_DUPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            point_list.clear()


cv.namedWindow(WINDOW_NAME)
img = cv.imread(IMAGE_PATH)
img_copy = img.copy()
org_image = img.copy()
cv.setMouseCallback(WINDOW_NAME, draw_distance)

LA2B = 15.96
LA1B = 11.3
ALPHA_SA = 75.8  # degrees
ALPHA_S = 286.11  # degrees
LATITUDE = 31.47
DATE = date(2021, 6, 12)

LA1A2 = hem.get_La1a2(LA2B, LA1B, ALPHA_S, ALPHA_SA)
delta = hem.compute_solar_declination(DATE)
omega = hem.get_solarhourangle(ALPHA_S, LATITUDE, delta, False)
h_s = hem.get_solarelevation(LATITUDE, delta, omega)
rhs, rcs = hem.get_ratios(h_s, ALPHA_S, ALPHA_SA, LA2B, LA1A2)

while True:
    cv.imshow(WINDOW_NAME, img)
    key = cv.waitKey(1) & 0xFF
    if key == ord("m"):
        mode = not mode
    elif key == ord("r"):
        img = org_image.copy()
        img_copy = org_image.copy()
    elif key == 27:
        break
