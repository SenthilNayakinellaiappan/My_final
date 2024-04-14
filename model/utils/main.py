import numpy as np
import torch
import cv2
from .cobb_evaluate import cobb_angle_calc


# Path to the TorchScript model file
model_scripted_path = 'model\models\model_scripted.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loaded_model = torch.jit.load(model_scripted_path) 
loaded_model.eval()

def prediction(image_path):
    input_tensor = preprocess(image_path)
    prediction = loaded_model(input_tensor.to(device))
    landmarks = identify_landmarks(image_path, input_tensor, prediction)
    cobb_angle = cobb_angle_calc(landmarks, input_tensor)
    pr_cobb_angles = np.asarray(cobb_angle, np.float32)
    return pr_cobb_angles

def preprocess(input_image):
    origin_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    target_height = 512
    target_width = 256
    resized_image = cv2.resize(origin_image, (target_width, target_height))
    image_tensor = torch.from_numpy(resized_image).unsqueeze(0).float()
    return image_tensor

def CV2plt(cv_img):
    b,g,r = cv2.split(cv_img)
    plt_img = cv2.merge([r,g,b]).astype(int)
    return plt_img

def identify_landmarks(path, tensor, prediction):
    batch, height, width = tensor.shape
    batch_img = np.zeros((height, width * batch, 3),int)
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
    cv2.imwrite(path, batch_img)
    return coord
