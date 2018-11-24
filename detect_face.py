import numpy as np
import cv2
import math


def detect_faces(f_cascade, colored_img, scale_factor=1.1):
    img_copy = np.copy(colored_img)
    # convert to gray
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=6)
    return faces


def crop_rot_images(frame, lbp_face_cascade, draw_face=False):
    # optimization req
    center, xc, yc, wc, hc = get_largest_face(frame, detect_faces(lbp_face_cascade, frame), return_face=True)
    left, xl, yl, wl, hl = get_largest_face(rotate_img(frame, 45),
                                            detect_faces(lbp_face_cascade, rotate_img(frame, 45)), return_face=True)
    right, xr, yr, wr, hr = get_largest_face(rotate_img(frame, -45),
                                             detect_faces(lbp_face_cascade, rotate_img(frame, -45)), return_face=True)
    x, y, z = center.shape
    p, q, r = left.shape
    temp = center
    if x * y < p * q:
        x, y, z = p, q, r
        temp = left
        xc, yc, wc, hc = xl, yl, wl, hl
    p, q, r = right.shape
    if x * y < p * q:
        x, y, z = p, q, r
        temp = right
        xc, yc, wc, hc = xr, yr, wr, hr
    if draw_face:
        cv2.rectangle(frame, (xc, yc), (wc, hc), (0, 255, 0), 2)
    return temp


def rotate_img(img, angle):
    num_rows, num_cols = img.shape[:2]
    rotation_mat = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_mat, (num_cols, num_rows))


def draw_faces(img, faces):
    # draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img


def draw_tilt_faces(frame, cascade):
    num_rows, num_cols = frame.shape[:2]
    center = detect_faces(cascade, frame)
    left = detect_faces(cascade, rotate_img(frame, 45))
    right = detect_faces(cascade, rotate_img(frame, -45))
    for (x, y, w, h) in center:
        # print((x, y, x + w, y + h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for (x, y, w, h) in left:
        (px, py) = rotate_point(x, y, num_cols, num_rows, -45)
        (qx, qy) = rotate_point(x + w, y, num_cols, num_rows, -45)
        (rx, ry) = rotate_point(x + w, y + h, num_cols, num_rows, -45)
        (sx, sy) = rotate_point(x, y + h, num_cols, num_rows, -45)
        (p, q, r, s) = (
            int(min(px, qx, rx, sx)), int(min(py, qy, ry, sy)), int(max(px, qx, rx, sx)), int(max(py, qy, ry, sy)))
        # print((p, q, r, s))
        cv2.rectangle(frame, (p, q), (r, s), (0, 255, 0), 2)
    for (x, y, w, h) in right:
        (px, py) = rotate_point(x, y, num_cols, num_rows, 45)
        (qx, qy) = rotate_point(x + w, y, num_cols, num_rows, 45)
        (rx, ry) = rotate_point(x + w, y + h, num_cols, num_rows, 45)
        (sx, sy) = rotate_point(x, y + h, num_cols, num_rows, 45)
        (p, q, r, s) = (
            int(min(px, qx, rx, sx)), int(min(py, qy, ry, sy)), int(max(px, qx, rx, sx)), int(max(py, qy, ry, sy)))
        # print("boo "+str((p, q, r, s)))
        cv2.rectangle(frame, (p, q), (r, s), (0, 255, 0), 2)
    return frame


def rotate_point(x, y, w, h, angle):
    x0 = math.cos(math.radians(angle)) * (x - w / 2) + math.sin(math.radians(angle)) * (y - h / 2)
    y0 = math.cos(math.radians(angle)) * (y - h / 2) - math.sin(math.radians(angle)) * (x - w / 2)
    x0 = x0 + w / 2
    y0 = y0 + h / 2
    return x0, y0


def get_largest_face(img, faces, draw_face=False, return_face=False):
    # draw and crop largest face
    largest_face = 0
    a = b = c = d = 0
    for (x, y, w, h) in faces:
        if (w * h) >= largest_face:
            largest_face = w * h
            a = x
            b = y
            c = w
            d = h
    if draw_face:
        cv2.rectangle(img, (a, b), (a + c, b + d), (0, 255, 0), 2)

    crop_img = img[b:b + d, a:a + c]
    if return_face:
        return crop_img, a, b, a + c, b + d
    else:
        return crop_img
