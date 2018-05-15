"""Class that encapsulates a TensorFlow Object Detection API data point"""
from object_detection.utils.dataset_util import bytes_list_feature
from object_detection.utils.dataset_util import float_list_feature
from object_detection.utils.dataset_util import int64_list_feature
from object_detection.utils.dataset_util import int64_feature
from object_detection.utils.dataset_util import bytes_feature


class TFDataPoint:
    def __init__(self):
        # Normalised (x/width, y/height) bounding box coordinates, origin: top left corner
        self.xMins = []
        self.xMaxs = []
        self.yMins = []
        self.yMaxs = []

        # Classes (text labels and corresponding integers)
        self.classLabels = []
        self.classes = []

        self.difficult = []         # Difficult flag
        self.image = None           # TF encoded image, .jpg or .png
        self.width = None           # image width
        self.height = None          # image height
        self.encoding = None        # encoding type
        self.filename = None        # image filename

    def create_data_point(self):
        return {
            "image/height": int64_feature(self.height),
            "image/width": int64_feature(self.width),
            "image/filename": bytes_feature(self.filename.encode("utf8")),
            "image/source_id": bytes_feature(self.filename.encode("utf8")),
            "image/encoded": bytes_feature(self.image),
            "image/format": bytes_feature(self.encoding.encode("utf8")),
            "image/object/bbox/xmin": float_list_feature(self.xMins),
            "image/object/bbox/xmax": float_list_feature(self.xMaxs),
            "image/object/bbox/ymin": float_list_feature(self.yMins),
            "image/object/bbox/ymax": float_list_feature(self.yMaxs),
            "image/object/class/text": bytes_list_feature(self.classLabels),
            "image/object/class/label": int64_list_feature(self.classes),
            "image/object/difficult": int64_list_feature(self.difficult)
        }
