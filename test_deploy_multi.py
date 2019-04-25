import argparse
import os, errno
import timeit
import glob
import xml.etree.ElementTree as elemTree

from face_detector import *
from tqdm import tqdm

from Box import BoxClass
from GroundTruth import LabelClass

IOU_THRESHOLD = 0.5
NO_OVERLAP = 0
count_image = 0
total_gt = 0
total_tp = 0
total_tn = 0
total_fp = 0
total_fn = 0
total_precision = 0
total_recall = 0
total_err = 0
total_fscore = 0
total_accuracy = 0
total_time = 0


def get_args():
    parser = argparse.ArgumentParser(
        description="This example script will show you how to use the face detector module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input_image_path", 
        "-ii",
        type=str, 
        required=True, 
        help="Path to image folder. ex) folder_name/class_name/file_name.jpg --> folder_name is required."
    )
    parser.add_argument(
        "--input_xml_path", 
        "-ig", 
        type=str, 
        required=True, 
        help="Path to xml folder containing the ground truth saved as PASCAL VOC format",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        default='mobilenetv2_ssdlite',
        help=
        "Please choose between : opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd, mobilenetv2_ssdlite "
    )
    parser.add_argument(
        "--model_type",
        "-mt",
        type=str,
        default="pb",
        help="Please choose : pb, tflite"
    )
    parser.add_argument(
        "--model_path",
        "--mp",
        type=str,
        default="models/ssd/frozen_inference_graph_face.pb",
        help="Path to model path. ex) /models/ssd/frozen_inference_face.pb"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Please output folder path. If it's not exist, it automatically creates it.")

    args = parser.parse_args()
    return args

def create_output_folder(input_path, output_path) :
    try :
        os.makedirs(output_path, exist_ok=True)
    except OSError as exc :
        raise

    try :
        filename_list = os.listdir(input_path)
        for filename in filename_list : 
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

def choose_face_detector(method, model_path, model_type) :
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
    elif (method == 'mobilenet_ssd') or (method == 'mobilenetv2_ssdlite'):
        if model_type == "pb":
            print("- TensoflowMobilNetSSDFaceDetector -")
            face_detector = TensoflowMobilNetSSDFaceDetector(
                det_threshold=0.75,
                model_path=model_path)
        elif model_type == "tflite":
            print("- TensorflowLiteMobileNetSSDFaceDetector -")
            face_detector = TensorflowLiteMobileNetSSDFaceDetector(
                det_threshold=0.75,
                model_path=model_path
            )

    return face_detector   

def processing_rect(image_data, faces, scores, bounding_box, rect_color, model_type, fd):
    
    set_color = rect_color
    image_data = image_data

    fontscale = 1.0
    thickness = 2

    axis_dict = {}
    
    idx = 0
    for face in faces:
        xmin = face[0]
        ymin = face[1]
        xmax = face[2]
        ymax = face[3]
        
        # print("- {0} Scores[{1}]:{2:.5f}:".format(model_type, idx, scores[idx]),"({0},{1},{2},{3})".format(xmin, ymin, xmax, ymax))

        axis_dict[idx] = [xmin, ymin, xmax, ymax]
        
        fd.write("- Scores[{0}]:{1:.5f}:".format(idx, scores[idx]) + " ({0},{1},{2},{3})".format(xmin, ymin, xmax, ymax))
        fd.write("\n")
        centerX = xmin + (int)((xmax - xmin) / 2)
        centerY = ymin + (int)((ymax - ymin) / 2)

        # draw rect
        cv2.rectangle(image_data, (xmin, ymin), (xmax, ymax), set_color, 4)
        # xmin, ymin
        #cv2.putText(image_data, "{0}, {1}".format(xmin, ymin), (xmin, ymin), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)
        # xmax, ymax
        #cv2.putText(image_data, "{0}, {1}".format(xmax, ymax), (xmax, ymax), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

        
        # idx
        if (model_type == "ssdlite") :
            cv2.putText(image_data, "{0}".format(idx), (xmin, ymin), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)
        elif (model_type == "ssd") :
            cv2.putText(image_data, "{0}".format(idx), (xmax, ymax), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)
        
        # scores
        cv2.putText(image_data, "{0:.5f}".format(scores[idx]), (xmin, ymax), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontscale, set_color, thickness, cv2.LINE_AA)

        idx = idx + 1

    bounding_box.axis = axis_dict

    return image_data, bounding_box

def get_filelist(input_path, type):
    
    file_list = None
    
    if type == 'all':
        file_list = sorted(glob.glob(input_path + "/*/*"))
    elif type == 'xml':
        file_list = sorted(glob.glob(input_path + "/*/*.xml"))

    return file_list

def get_gt_info(filename):
    
    tree = elemTree.parse(filename)
    root = tree.getroot()
    
    bb_count = len(root.findall("object"))
    # print(">> [DEBUG] bb_count : ", bb_count)
    
    bb_dict = {}
    idx = 0
    for bb in root.iter("bndbox"):
        bb_dict[idx] = [bb.findtext("xmin"), bb.findtext("ymin"), bb.findtext("xmax"), bb.findtext("ymax")]
        idx = idx + 1
    
    return bb_count, bb_dict


def intersection_over_union(box1, box2):
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))

    width = (x2 - x1)
    height = (y2 - y1)

    if (width < 0) or (height < 0):
        return NO_OVERLAP

    area_intersection = width * height
    # print("area_intersection = ", area_intersection)
    
    area_box1 = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    area_box2 = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))
    area_union = area_box1 + area_box2 - area_intersection

    iou = float(area_intersection / area_union)

    return iou

def processing_condition(gt_dict, bb_dict):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    # gt_dict_values = gt_dict.values()
    # gt_count = 0
    # gt_axis_dict = {}
    # for value in gt_dict_values:
    #     gt_count = value[0]
    #     gt_axis_dict = value[1]

    gt_axis_dict = gt_dict
    gt_count = len(gt_axis_dict.keys())
    # print("- gt_axis_dict : ", gt_axis_dict)
    # print("- gt_count : ", gt_count)
    # print("=================================")
    
    bb_count = len(bb_dict.keys())
    
    # print("-", bb_dict)
    if (gt_count is not 0) and (bb_count is 0):
        false_negative = gt_count
        return (true_positive, true_negative, false_positive, false_negative)
    
    if (gt_count is 0) and (bb_count is not 0):
        false_positive = bb_count
        return (true_positive, true_negative, false_positive, false_negative)
    
    idx_iou = 0
    iou_dict = {}
    no_overlap_count = 0
    true_positive_flag = False
    for key in bb_dict:
        # print(key)
        # print(bb_dict[key])

        for key_gt in gt_axis_dict:
            iou_dict[idx_iou] = intersection_over_union(bb_dict[key], gt_axis_dict[key_gt])
            # print("- idx_iou : ", idx_iou)
            # print("- iou_dict : ", iou_dict[idx_iou])
            # print("- key_gt : ", key_gt)
            # print("- gt_axis_dict : ", "[%d]".format(key_gt), " : ", gt_axis_dict[key_gt])
            # print("- bb : ", bb_dict[key])
            # print("-------")

            # True Positive
            if (iou_dict[idx_iou] >= IOU_THRESHOLD):
                true_positive = true_positive + 1
                true_positive_flag = True
            elif (iou_dict[idx_iou] == 0):
                no_overlap_count = no_overlap_count + 1

            idx_iou = idx_iou + 1

        if (no_overlap_count is not 0) and (true_positive_flag is False):
            false_positive = false_positive + 1

        false_negative = gt_count - true_positive
        
        idx_iou = 0
        true_positive_flag = False
    
    return (true_positive, true_negative, false_positive, false_negative)
    

def get_face(image_file_list, xml_file_list, output_path, face_detector):
    
    global count_image, total_gt, total_tp, total_tn, total_fp, total_fn
    global total_time
    
    
    log_filename = '/home/task1/Desktop/myungsung.kwak/project/FACE/face-detector-benchmark/detect_result_ssdlite_tflite.txt'
    face_detector = face_detector

    green = (0, 255, 0)
    cyan = (255, 255, 0)

    with open(log_filename, 'w', encoding='utf8') as f:
        
        idx = 0
        # Create label instance
        # label = LabelClass()
        bounding_box = BoxClass()
        
        for file_name in tqdm(image_file_list):
            
            print(">> [DEBUG] Image file : ", file_name)
            print(">> [DEBUG] XML file : ", xml_file_list[idx])
            
            # Get bounding box information from xml
            gt_count = 0
            gt_dict = {}
            gt_count, gt_dict = get_gt_info(xml_file_list[idx])
            idx = idx + 1
            
            f.write(file_name)
            f.write("\n")

            # Find face using face detector
            image_data = cv2.imread(file_name)

            # [Start] Check only inference time
            start_time = timeit.default_timer()

            # Detect face using method
            faces, scores = face_detector.detect_face(image_data)
            
            # [End] Check only inference time
            end_time = timeit.default_timer()
            
            
            total_time = (float)(total_time + (end_time - start_time))

            f.write("- Inference time(sec):{0:.5f}".format(end_time - start_time))
            f.write("\n")

            start_time = timeit.default_timer()
            end_time = timeit.default_timer()

            image_data, bounding_box = processing_rect(image_data, faces, scores, bounding_box, green, "ssdlite", f)
            # print(">> [DEBUG] gt_dict : ", gt_dict)
            # print(">> [DEBUG] bounding_box : ", bounding_box._axis_dict)

            # (true_positive, true_negative, false_positive, false_negative)
            predict_condition = processing_condition(gt_dict, bounding_box._axis_dict)
            # print("[DEBUG] predict_condition : ", predict_condition)
            
            f.write("- gt:" + str(gt_count))
            f.write("\n")
            f.write("- detect:" + str(len(faces)))
            f.write("\n")
            
            f.write("- tp:" + str(predict_condition[0]) 
                    + " tn:" + str(predict_condition[1]) 
                    + " fp:" + str(predict_condition[2]) 
                    + " fn:" + str(predict_condition[3]))
            f.write("\n")

            total_gt = total_gt + gt_count
            total_tp = total_tp + predict_condition[0]
            total_tn = total_tn + predict_condition[1]
            total_fp = total_fp + predict_condition[2]
            total_fn = total_fn + predict_condition[3]

            
            # Save file 
            fname, ext = os.path.splitext(file_name)
            fname = os.path.splitext(file_name)[0].split("/")
            class_name = fname[len(fname) - 2]
            fname = fname[len(fname) - 1]

            #label.set_gt_dict(fname+ext, gt_count, gt_dict)
            #print("[DEBUG] label._gt_dict : ", label._gt_dict)

            save_path = os.path.join(output_path, class_name)
            save_path = os.path.join(save_path, fname)
            save_path = save_path + ext

            f.write("- output:" + save_path)
            f.write("\n")
            cv2.imwrite(save_path, image_data)
            f.write("----------------------------------------------------------")
            f.write("\n")
            
            print(">>> [GT, TP, TN, FP, FN] : " , [total_gt, total_tp, total_tn, total_fp, total_fn])            
            print("====================================================================")

def main():
    global total_time
    global total_gt
    global total_precision, total_recall, total_err, total_fscore, total_accuracy
    
    args = get_args()

    create_output_folder(args.input_image_path, args.output)
    face_detector = choose_face_detector(args.method, args.model_path, args.model_type)

    if face_detector is not None :
        image_file_list = get_filelist(args.input_image_path, "all")
        xml_file_list = get_filelist(args.input_xml_path, "xml")
        get_face(image_file_list, xml_file_list, args.output, face_detector)
    else :
        print ('Please select the available method from this list: opencv_haar , dlib_hog , dlib_cnn , mtcnn , mobilenet_ssd, mobilenetv2_ssdlite')
        
    total_accuracy = (float)((total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn))
    total_precision = (float)(total_tp / (total_tp + total_fp))
    total_recall = (float)(total_tp / (total_tp + total_fn))
    total_err = (float)((total_fp + total_fn) / (total_tp + total_tn + total_fp + total_fn))
    total_fscore = (float)((2 * total_precision * total_recall) / (total_precision + total_recall))
    
    print("----------- SUMMARY -----------")
    print("[Ground truth] : ", total_gt)
    print("[Accuracy] : ", total_accuracy)
    print("[Error rate] : ", total_err)
    print("[Precision] : ", total_precision)
    print("[Recall] : ", total_recall)
    print("[F1 Score] : ", total_fscore)
    print(">> Total time : ", total_time)
    print("-------------------------------")

if __name__ == '__main__':
    main()
