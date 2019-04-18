import argparse
import cv2
import os, errno
import timeit
from face_detector import *


def get_args():
    parser = argparse.ArgumentParser(
        description=
        "This example script will show you how to use the face detector module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to image file")
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default='mobilenetv2_ssdlite',
        help=
        "Please choose between : opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd, mobilenetv2_ssdlite "
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Please output root path")

    args = parser.parse_args()
    return args

def create_output_folder(input_path, output_path) :
    try :
        os.makedirs(output_path, exist_ok=True)
    except OSError as exc :
        raise

    try :
        filenames = os.listdir(input_path)
        for filename in filenames : 
            try :
                out = os.path.join(output_path, filename)
                print(out)
                if not (os.path.isdir(out)) :
                    os.makedirs(out)
            except OSError as e :
                if e.errno != errno.EEXIST:
                    print("Failed to create directory!!!")
                    raise
    except PermissionError :
        pass

def choose_face_detector(method) :
    # Current available method in this repo
    method_list = ['opencv_haar', 'dlib_hog', 'dlib_cnn', 'mtcnn', 'mobilenet_ssd', 'mobilenetv2_ssdlite']

    face_detector = None

    # Initialize method
    if method == 'opencv_haar':
        face_detector = OpenCVHaarFaceDetector(
            scaleFactor=1.3,
            minNeighbors=5,
            model_path='models/haarcascade_frontalface_default.xml')
    elif method == 'dlib_hog':
        face_detector = DlibHOGFaceDetector(
            nrof_upsample=0, det_threshold=-0.2)

    elif method == 'dlib_cnn':
        face_detector = DlibCNNFaceDetector(
            nrof_upsample=0, model_path='models/mmod_human_face_detector.dat')

    elif method == 'mtcnn':
        face_detector = TensorflowMTCNNFaceDetector(model_path='models/mtcnn')

    elif method == 'mobilenet_ssd':
        face_detector = TensoflowMobilNetSSDFaceDector(
            det_threshold=0.75,
            model_path='models/ssd/frozen_inference_graph_face.pb')
    elif method == 'mobilenetv2_ssdlite':
        face_detector = TensoflowMobilNetV2SSDLiteFaceDector(
            det_threshold=0.75,
            model_path='models/ssdlite/trained_ssdlite_mobilenet_v2_233505.pb')

    return face_detector


def rect_visualization(image_data, faces, scores, rect_color):
    
    set_color = rect_color

    idx = 0
    for face in faces:
        xmin = face[0]
        ymin = face[1]
        xmax = face[2]
        ymax = face[3]

        print("- Scores[{0}]:{1:.5f}:".format(idx, scores[idx]),"({0},{1},{2},{3})".format(xmin, ymin, xmax, ymax))
        centerX = xmin + (int)((xmax - xmin) / 2)
        centerY = ymin + (int)((ymax - ymin) / 2)

        # draw rect
        cv2.rectangle(image_data, (xmin, ymin), (xmax, ymax), set_color,4)
        # xmin, ymin
        cv2.putText(image_data, "{0}, {1}".format(xmin, ymin), (xmin, ymin), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)
        # xmax, ymax
        cv2.putText(image_data, "{0}, {1}".format(xmax, ymax), (xmax, ymax), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)
        # scores
        cv2.putText(image_data, "{0:.5f}".format(scores[idx]), (xmin, ymax), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

        idx = idx + 1

def get_face_multi(input_path, output_path, _face_detector):
    
    # base_face_detector = "mobilenet_ssd"
    face_detector = _face_detector
    out_class_path = None

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)

    fontscale = 1.0
    thickness = 2

    try:
        filenames = os.listdir(input_path)
        for filename in filenames:
            full_filename = os.path.join(input_path, filename)
            if os.path.isdir(full_filename):
                out = os.path.join(output_path, filename)
                class_name = out.split('/')
                class_name = class_name[len(class_name)-1]
                out_class_path = os.path.join(output_path, class_name)
                try :
                    os.makedirs(out_class_path, exist_ok=True)
                except OSError as exc :
                    raise

                get_face(full_filename, output_path, face_detector)

            else:
                # ext = os.path.splitext(full_filename)[-1]
                print(full_filename)

                # Find face using face detector
                image_data = cv2.imread(full_filename)

                # Check only inference time
                start_time = timeit.default_timer()
                faces, scores = face_detector.detect_face(image_data)
                end_time = timeit.default_timer()
                print("- Inference time(sec) : {0:.5f}".format(end_time - start_time))

                rect_visualization(image_data, faces, scores, "green")

                
                # num of detections, highest score
                if len(faces) is not 0 :
                    print("- Num of dectect:", len(faces))
                    cv2.putText(image_data,
                                "Num : {0}, Top score : {1:.5f}".format(len(faces), scores[0]),
                                (50,50),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                2.0,
                                white,
                                3,
                                cv2.LINE_AA)
                    # Save file 
                    fname, ext = os.path.splitext(full_filename)
                    fname = os.path.splitext(full_filename)[0].split("/")
                    class_name = fname[len(fname)-2]
                    fname = fname[len(fname)-1]

                    save_path = os.path.join(output_path, class_name)
                    save_path = os.path.join(save_path, fname)
                    save_path = save_path + ext

                    print("- Output : ", save_path)
                    cv2.imwrite(save_path, image_data)
                    print("----------------------------------------------------------")


    except PermissionError:
        pass


def get_face(input_path, output_path, _face_detector):

    face_detector = _face_detector
    out_class_path = None

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    yellow = (0, 255, 255)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)

    fontscale = 1.0
    thickness = 2

    try:
        filenames = os.listdir(input_path)
        for filename in filenames:
            full_filename = os.path.join(input_path, filename)
            if os.path.isdir(full_filename):
                out = os.path.join(output_path, filename)
                class_name = out.split('/')
                class_name = class_name[len(class_name)-1]
                out_class_path = os.path.join(output_path, class_name)
                try :
                    os.makedirs(out_class_path, exist_ok=True)
                except OSError as exc :
                    raise

                get_face(full_filename, output_path, face_detector)

            else:
                # ext = os.path.splitext(full_filename)[-1]
                print(full_filename)

                # Find face using face detector
                image_data = cv2.imread(full_filename)

                # Check only inference time
                start_time = timeit.default_timer()
                faces, scores = face_detector.detect_face(image_data)
                end_time = timeit.default_timer()
                print("- Inference time(sec) : {0:.5f}".format(end_time - start_time))

                idx = 0
                for face in faces :
                    if (idx == 0) :
                        set_color = green
                    elif (idx == 1) :
                        set_color = cyan
                    elif (idx == 2) :
                        set_color = yellow
                    else :
                        set_color = red
                
                    # Resulting detection will be in numpy array format
                    # E.g :
                    #
                    # [[xmin,ymin,xmax,ymax]
                    #           ...
                    #  [xmin,ymin,xmax,ymax]]

                    xmin = face[0]
                    ymin = face[1]
                    xmax = face[2]
                    ymax = face[3]

                    print("- Scores[{0}]:{1:.5f}:".format(idx, scores[idx]),"({0},{1},{2},{3})".format(xmin, ymin, xmax, ymax))

                    centerX = xmin + (int)((xmax - xmin) / 2)
                    centerY = ymin + (int)((ymax - ymin) / 2)

                    # Usage : cv2.rectangle( image_data,(xmin,ymin),(xmax,ymax),colour in BGR format : e.g (255,0,0) , line thickness = e.g 2 )

                    # Draw detected box
                    cv2.rectangle(image_data,
                                  (xmin, ymin),
                                  (xmax, ymax),
                                  set_color,
                                  4)

                    # xmin, ymin
                    cv2.putText(image_data, 
                                "{0}, {1}".format(xmin, ymin),
                                (xmin, ymin),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                fontscale,
                                set_color,
                                thickness,
                                cv2.LINE_AA)
                    # xmax, ymax
                    cv2.putText(image_data, 
                                "{0}, {1}".format(xmax, ymax),
                                (xmax, ymax),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                                fontscale, 
                                set_color, 
                                thickness, 
                                cv2.LINE_AA)
                    # scores
                    cv2.putText(image_data,
                                "{0:.5f}".format(scores[idx]),
                                (xmin, ymax),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                fontscale,
                                set_color,
                                thickness,
                                cv2.LINE_AA)
                    idx = idx + 1
                
                # num of detections, highest score
                if len(faces) is not 0 :
                    print("- Num of dectect:", len(faces))
                    cv2.putText(image_data,
                                "Num : {0}, Top score : {1:.5f}".format(len(faces), scores[0]),
                                (50,50),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                2.0,
                                white,
                                3,
                                cv2.LINE_AA)
                    # Save file 
                    fname, ext = os.path.splitext(full_filename)
                    fname = os.path.splitext(full_filename)[0].split("/")
                    class_name = fname[len(fname)-2]
                    fname = fname[len(fname)-1]

                    save_path = os.path.join(output_path, class_name)
                    save_path = os.path.join(save_path, fname)
                    save_path = save_path + ext

                    print("- Output : ", save_path)
                    cv2.imwrite(save_path, image_data)
                    print("----------------------------------------------------------")

    except PermissionError:
        pass


def main():
    args = get_args()

    create_output_folder(args.input, args.output)
    face_detector = choose_face_detector(args.method)

    if face_detector is not None :
        # get_face(args.input, args.output, face_detector)
        get_face_multi(args.input, args.output, face_detector)
    else :
        print ('Please select the available method from this list: opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd')

    # Read image file using opencv
    # image_data = cv2.imread(args.input)

    # # Current available method in this repo
    # method_list = [
    #     'opencv_haar', 'dlib_hog', 'dlib_cnn', 'mtcnn', 'mobilenet_ssd', 'mobilenetv2_ssdlite'
    # ]
    # method = args.method

    # # Initialize method
    # if method == 'opencv_haar':
    #     face_detector = OpenCVHaarFaceDetector(
    #         scaleFactor=1.3,
    #         minNeighbors=5,
    #         model_path='models/haarcascade_frontalface_default.xml')

    # elif method == 'dlib_hog':
    #     face_detector = DlibHOGFaceDetector(
    #         nrof_upsample=0, det_threshold=-0.2)

    # elif method == 'dlib_cnn':
    #     face_detector = DlibCNNFaceDetector(
    #         nrof_upsample=0, model_path='models/mmod_human_face_detector.dat')

    # elif method == 'mtcnn':
    #     face_detector = TensorflowMTCNNFaceDetector(model_path='models/mtcnn')

    # elif method == 'mobilenet_ssd':
    #     face_detector = TensoflowMobilNetSSDFaceDector(
    #         det_threshold=0.3,
    #         model_path='models/ssd/frozen_inference_graph_face.pb')
    # elif method == 'mobilenetv2_ssdlite':
    #     face_detector = TensoflowMobilNetV2SSDLiteFaceDector(
    #         det_threshold=0.3,
    #         model_path='models/ssdlite/trained_ssdlite_mobilenet_v2_233505.pb')

    # if method not in method_list:
    #     #print 'Please select the available method from this list: opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd'
    #     print ('Please select the available method from this list: opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd')

    # else:


    #     image_data = cv2.imread(args.input)

    #     # Detect face in the image
    #     detected_face = face_detector.detect_face(image_data)

    #     # Resulting detection will be in numpy array format
    #     # E.g :
    #     #
    #     # [[xmin,ymin,xmax,ymax]
    #     #             ...
    #     #  [xmin,ymin,xmax,ymax]]

    #     # Visualizing detected face
    #     red = (0, 0, 255)
    #     green = (0, 255, 0)
    #     blue = (255, 0, 0)
    #     white = (255, 255, 255)
    #     yellow = (0, 255, 255)
    #     cyan = (255, 255, 0)
    #     magenta = (255, 0, 255)

    #     fontscale = 1.0
    #     thickness = 2

    #     faces, scores = detected_face

    #     idx = 0
    #     for face in faces:


    #         if (idx ==0)  :
    #             set_color = green
    #         elif (idx == 1) :
    #             set_color = cyan
    #         elif (idx == 2) :
    #             set_color = yellow
    #         else :
    #             set_color = red

    #         # Usage : cv2.rectangle( image_data,(xmin,ymin),(xmax,ymax),colour in BGR format : e.g (255,0,0) , line thickness = e.g 2 )
    #         cv2.rectangle(image_data, 
    #             (face[0], face[1]), (face[2], face[3]),
    #             set_color, 4)

    #         xmin = face[0]
    #         ymin = face[1]
    #         xmax = face[2]
    #         ymax = face[3]

    #         centerX = xmin + (int)((xmax - xmin) / 2)
    #         centerY = ymin + (int)((ymax - ymin) / 2)

    #         cv2.putText(image_data, "{0}, {1}".format(xmin, ymin), (face[0], face[1]), 
    #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

    #         cv2.putText(image_data, "{0}, {1}".format(xmax, ymax), (face[2], face[3]), 
    #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

    #         cv2.putText(image_data, "{0:.5f}".format(scores[idx]), (xmin, ymax),
    #             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

    #         idx = idx + 1
    #         # print("- 0 ---------")
    #         # print(face[0])
    #         # print("- 1 ---------")
    #         # print(face[1])
    #         # print("- 2 ---------")
    #         # print(face[2])
    #         # print("- 3 ---------")
    #         # print(face[3])

    #     cv2.putText(image_data, "Num : {0}, rank1 : {1:.5f}".format(idx, scores[0]),(50,50),
    #         cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.0, white, 3, cv2.LINE_AA)

    #     # Save image
    #     # cv2.imwrite('output', image_data)
    #     #print("------")
    #     #print(image_data)
    #     filename=os.path.splitext(args.input)[0].split("/")[2] # [images, jhk100, filename]
        
    #     save_path = args.output + "/" + filename + ".jpg"
    #     cv2.imwrite(save_path, image_data)
    #     print("- Save complte -")
    #     #cv2.imshow('Face Detection', image_data)
    #     #cv2.waitKey(0)


if __name__ == '__main__':
    main()
