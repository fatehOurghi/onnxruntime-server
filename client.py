from processing import *
import cv2
import numpy as np
import requests
import json
import os

url = "http://127.0.0.1:8001"
input_shape = {
    "batch_size": 1,
    "height": 768,
    "width": 1024,
    "depth": 3,
    "order": "bhwd"  #"bdhw"
}
pre_order = "bhwd"

def ort_client(image, model, dtype, pre_order, input_shape, url):
    # preprocess
    img = preprocess(image, input_shape, pre_order)

    # prepare data
    data_to_send = {
        "input": img.tolist(),
        "model": model,
        "dtype": dtype
    }
    headers = {
        'content-type': 'application/json'
    }

    # send to server and receive response
    response = requests.post(url, json=data_to_send, headers=headers)

    # postprocess
    data = json.loads(response.text)
    densitymap = postprocess(np.array(data['output']))
    count = int(np.sum(densitymap))

    return densitymap * 255, count



# load the image
img = cv2.imread("img.jpg")

densitymap, count = ort_client(img, "model.onnx", "float32", pre_order, input_shape, url)

print("count", count)

cv2.imshow("density map", densitymap*255)
cv2.waitKey(0)
