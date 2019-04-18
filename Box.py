class BoxClass:
    def __init__(self):
        self._axis_dict = {}
    
    @property
    def axis(self):
        return self._axis_dict
    
    @axis.setter
    def axis(self, new_axis):
        self._axis_dict = new_axis