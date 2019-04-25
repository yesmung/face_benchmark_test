import numpy as np
import cv2
import dlib
import resources.mtcnn.mtcnn as mtcnn
import tensorflow as tf

## Initializer for face detector classes


class OpenCVHaarFaceDetector():
    def __init__(self,
                 scaleFactor=1.3,
                 minNeighbors=5,
                 model_path='models/haarcascade_frontalface_default.xml'):

        self.face_cascade = cv2.CascadeClassifier(model_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, self.scaleFactor, self.minNeighbors)
        faces = [[x, y, x + w, y + h] for x, y, w, h in faces]

        return np.array(faces)


class DlibHOGFaceDetector():
    def __init__(self, nrof_upsample=0, det_threshold=0):
        self.hog_detector = dlib.get_frontal_face_detector()
        self.nrof_upsample = nrof_upsample
        self.det_threshold = det_threshold

    def detect_face(self, image):

        dets, scores, idx = self.hog_detector.run(image, self.nrof_upsample,
                                                  self.det_threshold)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.left())
            y1 = int(d.top())
            x2 = int(d.right())
            y2 = int(d.bottom())

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)


class DlibCNNFaceDetector():
    def __init__(self,
                 nrof_upsample=0,
                 model_path='models/mmod_human_face_detector.dat'):

        self.cnn_detector = dlib.cnn_face_detection_model_v1(model_path)
        self.nrof_upsample = nrof_upsample

    def detect_face(self, image):

        dets = self.cnn_detector(image, self.nrof_upsample)

        faces = []
        for i, d in enumerate(dets):
            x1 = int(d.rect.left())
            y1 = int(d.rect.top())
            x2 = int(d.rect.right())
            y2 = int(d.rect.bottom())
            score = float(d.confidence)

            faces.append(np.array([x1, y1, x2, y2]))

        return np.array(faces)


class TensorflowMTCNNFaceDetector():
    def __init__(self, model_path='models/mtcnn'):

        self.minsize = 15
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            with self.sess.as_default():
                self.pnet, self.rnet, self.onet = mtcnn.create_mtcnn(
                    self.sess, model_path)

    def detect_face(self, image):

        dets, face_landmarks = mtcnn.detect_face(
            image, self.minsize, self.pnet, self.rnet, self.onet,
            self.threshold, self.factor)

        faces = dets[:, :4].astype('int')
        conf_score = dets[:, 4]

        return faces


class TensoflowMobilNetSSDFaceDetector():
    def __init__(self,
                 det_threshold=0.75,
                 model_path='models/ssd/frozen_inference_graph_face.pb'):

        self.det_threshold = det_threshold
        self.model_path = model_path
        
        print("model_path : ", self.model_path)
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)

    def detect_face(self, image):

        h, w, c = image.shape

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        
        image_tensor = self.detection_graph.get_tensor_by_name(
            'image_tensor:0')

        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        
        # print("-----")
        # print("[TF face_detect] box shape : ", boxes.shape)
        # print("[TF face_detect] score  : ", scores)
        # print("[TF face_detect] class  : ", classes)
        # print("[TF face_detect] num_dectection : ", num_detections)

        filtered_score_index = np.argwhere(
            scores >= self.det_threshold).flatten()

        selected_boxes = boxes[filtered_score_index]
        selected_scores = scores[filtered_score_index]

        faces = np.array([[
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        ] for y1, x1, y2, x2 in selected_boxes])
        # print("[TF face_detect] face : ", faces)
        # print("[TF face_detect] scores : ", selected_scores)
        # print("-----")

        return faces, selected_scores

class TensorflowLiteMobileNetSSDFaceDetector():
    def __init__(self,
                 det_threshold=0.75,
                 model_path='models/ssdlite/trained_ssdlite_mobilenet_v2_414114.tflite'):
        
        self.det_threshold = det_threshold
        self.model_path = model_path
        self.detection_graph = tf.Graph()
        
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def detect_face(self, image):
        with self.detection_graph.as_default():
            with tf.Session() as sess:

                h, w, c = image.shape
                
                image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_np = tf.image.resize_images(image, size=[512, 512], method=tf.image.ResizeMethod.BILINEAR)

                resize_image = tf.image.resize_images(image_np, size=[512, 512], method=tf.image.ResizeMethod.BILINEAR)
                resize_image = (resize_image - 128.0) / 128.0
                resize_image = tf.expand_dims(resize_image, 0)
                
                # input_data = tf.summary.image("input_data", resize_image, max_outputs=3)
                input_data = sess.run(resize_image)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
                classes = self.interpreter.get_tensor(self.output_details[1]['index'])
                scores = self.interpreter.get_tensor(self.output_details[2]['index'])
                num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)

                # print("-----")
                # print("[TFLite face_detect] box shape : ", boxes.shape)
                # print("[TFLite face_detect] score  : ", scores)
                # print("[TFLite face_detect] class  : ", classes)
                # print("[TFLite face_detect] num_dectection : ", num_detections)

                filtered_score_index = np.argwhere(
                    scores >= self.det_threshold).flatten()

                selected_boxes = boxes[filtered_score_index]
                selected_scores = scores[filtered_score_index]

                faces = np.array([[
                    int(x1 * w),
                    int(y1 * h),
                    int(x2 * w),
                    int(y2 * h),
                ] for y1, x1, y2, x2 in selected_boxes])
                
                # print("[TFLite face_detect] face : ", faces)
                # print("[TFLite face_detect] scores : ", selected_scores)
                # print("-----")

                return faces, selected_scores