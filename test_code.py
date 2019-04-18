from typing import Dict, List, Union

from Box import BoxClass
from GroundTruth import LabelClass

import os, sys

# - ssdlite Scores[0]:0.99466: (505,860,890,1313)
# - ssdlite Scores[1]:0.97721: (0,752,471,1454)
# - ssdlite Scores[2]:0.91111: (1059,429,1491,1334)
# - ssd Scores[0]:0.99886: (1043,486,1474,1262)
# - ssd Scores[1]:0.97783: (500,896,904,1316)
# - ssd Scores[2]:0.97431: (0,801,458,1426)

# ssdlite_box = BoxClass()
# label = LabelClass()
# 
# ssdlite_box_list = {}
# ssdlite_box.axis = [505,860,890,1313]
# ssdlite_box_list[0] = ssdlite_box.axis
# ssdlite_box.axis = [0,752,471,1454]
# ssdlite_box_list[1] = ssdlite_box.axis
# print(ssdlite_box_list)
# print("-------------------------------")
# bb_count = 3
# bb_dict = {} # type: Dict[int, List[int]]
# bb_dict[0] = [1000, 2000, 1500, 2500]
# bb_dict[1] = [1001, 2000, 1500, 2500]
# bb_dict[2] = [1002, 2000, 1500, 2500]
# print(bb_dict)
# label.set_gt_dict("face001.jpg", bb_count, bb_dict)
# print(label._gt_dict)
# 
# ssdlite_box_list = {}
# ssdlite_box.axis = []
# ssdlite_box.axis = [100, 200, 300, 400]
# ssdlite_box_list[0] = ssdlite_box.axis
# ssdlite_box.axis = [101, 201, 301, 401]
# ssdlite_box_list[1] = ssdlite_box.axis
# bb_count = 2
# bb_dict = {}
# bb_dict[0] = [1111, 1111, 1111, 1111]
# bb_dict[2] = [2222, 2222, 2222, 2222]
# label.set_gt_dict("face002.jpg", bb_count, bb_dict)
# print(label._gt_dict)
# 
# filename = '/home/task1/Desktop/myungsung.kwak/project/FACE/face-detector-benchmark/detect_result_test.jpg'
# fname, ext = os.path.splitext(filename)
# print("0 :", fname)
# fname = os.path.splitext(filename)[0].split("/")  # type: Union[List[bytes], List[str]]
# 
# print("1 : ", fname)
# # class_name = fname[len(fname) - 2]
# fname = fname[len(fname) - 1]
# fname = fname + str(".xml")
# print("- ", fname)

def intersection_over_union(box1, box2):
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))
    
    width = (x2 - x1)
    height = (y2 - y1)
    
    if (width < 0 ) or (height < 0):
        return 0

    area_intersection = width * height
    print("area_intersection = ", area_intersection)
    #area_box1 = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    #area_box2 = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))
    area_box1 = (int(box1[2]) - int(box1[0])) * (int(box1[3]) - int(box1[1]))
    area_box2 = (int(box2[2]) - int(box2[0])) * (int(box2[3]) - int(box2[1]))
    area_union = area_box1 + area_box2 - area_intersection

    iou = float(area_intersection / area_union)

    return iou

bb_dict = {0: [505, 860, 890, 1313], 1: [0, 752, 471, 1454], 2: [1059, 429, 1491, 1334], 3: [20, 80, 300, 400], 4: [10, 100, 150, 200]}
gt_dict = {'face064.jpg': (3, {0: ['24', '845', '501', '1500'], 1: ['515', '898', '921', '1309'], 2: ['1009', '474', '1491', '1358']})}

gt_dict_values = gt_dict.values()
print(type(gt_dict_values))

gt_count = 0
gt_axis_dict = {}
for value in gt_dict_values:
    # print("value : ", value)
    # print("value[0] : ", value[0])
    # print("value[1] : ", value[1])
    gt_count = value[0]
    gt_axis_dict = value[1]

print(gt_axis_dict)
print(len(gt_axis_dict.keys()))
print("=================================")

IOU_THRESHOLD = 0.5

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

idx_iou = 0
iou_dict = {}
no_overlap_count = 0
true_positive_flag = False

for key in bb_dict:
    print(key)
    #print(bb_dict[key])
    
    for key_gt in gt_axis_dict:
        iou_dict[idx_iou] = intersection_over_union(bb_dict[key], gt_axis_dict[key_gt])
        print("- idx_iou : ", idx_iou)
        print("- iou_dict : ", iou_dict[idx_iou])
        print("- key_gt : ", key_gt)
        print("- gt_axis_dict : ", "[%d]".format(key_gt), " : ", gt_axis_dict[key_gt])
        print("- bb : ", bb_dict[key])
        print("-------")
        
        # True Positive
        if (iou_dict[idx_iou] >= IOU_THRESHOLD):
            true_positive = true_positive + 1
            true_positive_flag = True
        elif (iou_dict[idx_iou] == 0):
            no_overlap_count = no_overlap_count + 1
        # False Positive
        # False Negative

        idx_iou = idx_iou + 1
    
    if (no_overlap_count is not 0) and (true_positive_flag is False):
        false_positive = false_positive + 1
    idx_iou = 0
    true_positive_flag = False
    
print("true positive : ", true_positive)
false_negative = gt_count - true_positive
print("false negative : ", false_negative)
print("false positive : ", false_positive)
