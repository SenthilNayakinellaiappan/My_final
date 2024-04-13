import numpy as np
import torch
import cv2
from model.models import spinal_net
from .cobb_evaluate import cobb_angle_calc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = spinal_net.ConvNet()
model.load_state_dict(torch.load("model/models/shallow.pth", map_location='cpu'))

def prediction(image_path):
    input_tensor = preprocess(image_path)
    prediction = model(input_tensor.to(device))
    landmarks = identify_landmarks(image_path, input_tensor, prediction)
    cobb_angle = cobb_angle_calc(landmarks, input_tensor)
    pr_cobb_angles = np.asarray(cobb_angle, np.float32)
    return pr_cobb_angles


def preprocess(input_image):
    origin_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    target_height = 512
    target_width = 256
    image = np.zeros((target_height, target_width), np.uint8)
    cv2.resize(origin_image, (target_width, target_height), image)
    image = np.reshape(image, (1, image.shape[0], image.shape[1]))
    image_tensor = torch.from_numpy(image).float()
    return image_tensor

def CV2plt(cv_img):
    b,g,r = cv2.split(cv_img)
    plt_img = cv2.merge([r,g,b]).astype(np.int)
    return plt_img

def identify_landmarks(path, tensor, prediction):
    batch, height, width = tensor.shape
    batch_img = np.zeros((height, width * batch, 3), np.int)
    coord = []

    for i in range(batch):
        sample = tensor.numpy().squeeze(0)
        sample_BGR = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)
        predict = prediction[i]
        point_num = 68
        for j in range(point_num):
            cv2.circle(sample_BGR, (int(predict[j] * width), int(predict[j + point_num] * height)), 3, (255, 0, 0))
            coord.append([int(predict[j] * width), int(predict[j + point_num] * height)])
        plt_img = CV2plt(sample_BGR)
        batch_img[:, i * width:(i + 1) * width, :] = plt_img
    cv2.imwrite(path,batch_img)
    return coord