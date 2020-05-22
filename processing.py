import numpy as np
import cv2
import onnxruntime as rt


########################################################################################################################
#################################################   Processing   #######################################################
########################################################################################################################

def norm_rgb_img(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(0, 3):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img


def isgray(img):
    if len(img.shape) == 3:
        return False
    return True


def preprocess(img, input_shape, pre_order):
    w = input_shape["width"] if isinstance(input_shape["width"], int) and input_shape["width"] > 0 else img.shape[0]
    h = input_shape["height"] if isinstance(input_shape["height"], int) and input_shape["height"] > 0 else img.shape[1]
    img = cv2.resize(img, (w, h))
    # rgb required
    if input_shape["depth"] == 3:
        if isgray(img):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = img / 255.0
        img = norm_rgb_img(img)
    # gray required
    else:
        if not isgray(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img / 255.0
        img = np.reshape(img, (w, h, 1))
    img = np.expand_dims(img.astype(np.float32), axis=0)
    return np.einsum(pre_order + '->' + input_shape["order"], img)


def postprocess(img):
    return np.squeeze(img)


########################################################################################################################
#################################################   Inference   ########################################################
########################################################################################################################


#input_shape = {
#    "batch_size": 1,
#    "height": 768,
#    "width": 1024,
#    "depth": 1,
#    "order": "bdhw"  # "bdhw"
#}
#pre_order = "bhwd"
#
#img = cv2.imread("img.jpg")
#im = preprocess(img, input_shape, pre_order)
#
#session = rt.InferenceSession("models/model.onnx")
#
#out = session.run(None, {session.get_inputs()[0].name: im})
#print(np.sum(out[0]))
#cv2.imshow("dm", postprocess(out[0]) * 255)
#cv2.waitKey(0)
