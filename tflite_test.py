import numpy as np
import tensorflow as tf

import cv2
from PIL import  Image

file_name = "/home/task1/Desktop/myungsung.kwak/project/DataShare/test/face006/face017.jpg"
image_data = cv2.imread(file_name)
image_data = cv2.resize(image_data, 512, 512)

h, w, c = image_data.shape
image_np = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)



image_np_expanded = np.expand_dims(image_np, axis=0)

interpreter = tf.lite.Interpreter(model_path="models/ssdlite/trained_ssdlite_mobilenet_v2_414114.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details() # input image
output_details = interpreter.get_output_details() # box, score, class, num detections

print("===== Input details =====")
for name in input_details:
    print(name)

print("===== Output details =====")
for name in output_details:
    print(name)

input_shape = input_details[0]['shape']

sample_data = np.random.random_sample(input_shape)
print("====================")
print(type(sample_data))
print(sample_data.shape)
print("====================")


# resized_data = np.resize(image_np_expanded, (1, 512, 512, 3))
input_data = np.array(image_np_expanded, dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

boxes = interpreter.get_tensor(output_details[0]['index'])
scores = interpreter.get_tensor(output_details[1]['index'])
classes = interpreter.get_tensor(output_details[2]['index'])
num_detections = interpreter.get_tensor(output_details[3]['index'])

print(boxes)
print("-----")
print(scores)
print("-----")
print(classes)
print("-----")
print(num_detections)
print("-----")