import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
from io import BytesIO
import string
import imutils

app = Flask(__name__)

def normalization(img):
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def convert_to_char(prediction):
    alphabets = string.ascii_uppercase
    alphabets_low = string.ascii_lowercase
    numbers = string.digits
    character_mapping = numbers + alphabets + alphabets_low

    class_index = np.argmax(prediction)
    predicted_char = character_mapping[class_index]

    return predicted_char

def enhance_image(img):
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, h=5)
    image_gray = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2GRAY)
    min_val, max_val, _, _ = cv2.minMaxLoc(image_gray)
    threshold_value = int((min_val + max_val) / 2)
    ret, threshed = cv2.threshold(image_gray, threshold_value, 255, cv2.THRESH_TRUNC)
    _, binary_img = cv2.threshold(threshed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel2 = np.ones((1, 2), np.uint8)
    smoothed_img = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, kernel2)
    _, segmented_img = cv2.threshold(smoothed_img, 1, 255, cv2.THRESH_BINARY)
    return segmented_img

def predict_char(roi, model, texts_roi):
    merged_char_tmp = find_merged_contour(roi, texts_roi)
    sentence = ""
    for i, contour in enumerate(merged_char_tmp):
        x, y, w, h = cv2.boundingRect(contour)
        if h >= 15:
            tmp_h = h
            tmp_w = w

            if h > w:
                tmp_h = h + 20
                tmp_w = w + 20 + (h - w)
            else:
                tmp_w = w + 20
                tmp_h = h + 20 + (w - h)

            try:
                white_rect = np.ones(((tmp_h), tmp_w), dtype=np.uint8) * 255
                x_offset = max(0, (white_rect.shape[1] - w) // 2)
                y_offset = max(0, (white_rect.shape[0] - h) // 2)
                white_rect[y_offset:y_offset + h, x_offset:x_offset + w] = contour
                digit_resized = cv2.resize(white_rect, (28, 28))
                digit_resized = digit_resized.astype('float32') / 255.0
                digit_resized = np.expand_dims(digit_resized, axis=-1)
                digit_resized = np.expand_dims(digit_resized, axis=0)
                prediction = model.predict(digit_resized)
                predicted_char = convert_to_char(prediction)
                if predicted_char == "1":
                    predicted_char = "I"
                elif predicted_char == "0":
                    predicted_char = "O"
                elif predicted_char == "5":
                    predict_char = "S"
                elif predicted_char == "4":
                    predicted_char = "A"
                elif predicted_char == "8" or predicted_char == "3":
                    predicted_char = "B"
                sentence = sentence + predicted_char
            except Exception as e:
                print(e)
                sentence = sentence + "?"
    return sentence

def find_avg_contour(texts_roi):
    count = 0
    sum = 0
    for idx, roi in enumerate(texts_roi):
        cropped_img = texts_roi[idx]
        _, thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cropped_img = texts_roi[idx]
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            digit_img = cropped_img[y:y + h, x:x + w]
            count = count + 1
            sum = sum + digit_img.shape[0]
    try:
        return sum / count
    except Exception as e:
        print(e)

def find_merged_contour(roi, texts_roi):
    avg = find_avg_contour(texts_roi)
    merged_char_tmp = []
    cropped_img = roi
    _, thresh = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        digit_img_tmp = cropped_img[y:y + h, x:x + w]
        if w > (avg + 30):
            half_width = w // 2
            left_part = digit_img_tmp[:, :half_width]
            right_part = digit_img_tmp[:, half_width:]
            merged_char_tmp.append(left_part)
            merged_char_tmp.append(right_part)
        else:
            merged_char_tmp.append(digit_img_tmp)
    return merged_char_tmp

def super_function(img, texts_roi, model_upper_number, segmented_img):
    img2 = img.copy()
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
    height, width = img.shape[:2]
    result = ""
    futures = []
    for contour in conts:
        contour_tmp = []
        X, Y, W, H = cv2.boundingRect(contour)
        if 10 < H < 100 and 20 < W < 1500 and 2 < X < int(width * 0.95) and 2 < Y < int(height * 0.95):
            contour_tmp.append(contour)
            for idx, c in enumerate(contour_tmp):
                (x, y, w, h) = cv2.boundingRect(c)
                if (w >= 30 and w <= 1000) and (h >= 15 and h <= 100):
                    roi = segmented_img[y:y + h, x:x + w]
                    if X > (width * 0.18):
                        futures.append(executor.submit(predict_char, roi, model_upper_number, texts_roi))
                    else:
                        futures.append(executor.submit(predict_char, roi, model_upper_number, texts_roi))
                else:
                    continue
    for future in futures:
        result += future.result() + " "
    return result

def rlsa_horizontal(img, threshold):
    result = img.copy()
    height, width = img.shape[:2]
    for y in range(height):
        count = 0
        for x in range(width):
            if img[y, x] == 255:
                count += 1
            else:
                if count <= (threshold + 10):
                    result[y, x - count:x] = 0
                else:
                    if count <= threshold:
                        result[y, x - count:x] = 0
                count = 0
        if count <= threshold:
            result[y, width - count:width] = 0
    return result

def rlsa_vertical(img, threshold):
    result = img.copy()
    height, width = img.shape[:2]
    for x in range(width):
        count = 0
        for y in range(height):
            if img[y, x] == 255:
                count += 1
            else:
                if count <= threshold:
                    result[y - count:y, x] = 0
                count = 0
        if count <= threshold:
            result[height - count:height, x] = 0
    return result

def perform_rlsa(image, horizontal_threshold=2, vertical_threshold=5):
    rlsa_horizontal_result = rlsa_horizontal(image, threshold=horizontal_threshold)
    rlsa_result = rlsa_vertical(rlsa_horizontal_result, threshold=vertical_threshold)
    rlsa_combined = 255 - rlsa_result
    return rlsa_combined

def find_contours(img, img3):
    img2 = img3.copy()
    contours_in_area = []
    conts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    conts = imutils.grab_contours(conts)
    conts = sorted(conts, key=lambda x: (cv2.boundingRect(x)[1], cv2.boundingRect(x)[0]))
    for contour in conts:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        contours_in_area.append(contour)
    return contours_in_area

def resize_img(img):
    height, width = img.shape[:2]
    if width < 1500:
        new_width = width * 2
        new_height = height * 2
        img = cv2.resize(img, (new_width, new_height))
    return img

def histogram_equalization_color(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=1.3, tileGridSize=(2, 2))
    yuv[:,:,0] = clahe.apply(yuv[:,:,0])
    equalized_image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    return equalized_image

def find_roi(contour, segmented_img):
    texts_roi = []
    for idx, c in enumerate(contour):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = segmented_img[y:y+h, x:x+w]
        texts_roi.append(roi)
    return texts_roi

model_upper = load_model('model_ocr_number_upper.h5', compile=False)

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    base64_img = data['image']
    decoded_img = base64.b64decode(base64_img)
    img = Image.open(BytesIO(decoded_img))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = resize_img(img)
    equal_img = histogram_equalization_color(img)
    segmented_img = enhance_image(equal_img)
    rlsa_img = perform_rlsa(segmented_img)
    conts = find_contours(rlsa_img, img)
    texts_roi = find_roi(conts, segmented_img)
    sentence = super_function(rlsa_img, texts_roi, model_upper, segmented_img)
    return jsonify({'recognized_text': sentence})

@app.route('/')
def index():
    return "hellow"