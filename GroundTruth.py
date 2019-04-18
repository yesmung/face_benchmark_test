from typing import Dict, Any


class LabelClass:
    
    _gt_dict = ...  # type: Dict[Any, Any]

    def __init__(self):
        self._filename_list = []
        self._gt_dict = {}
    
    def set_gt_dict(self, filename, gt_count, gt_dict):
        self._gt_dict = {filename : (gt_count, gt_dict)}