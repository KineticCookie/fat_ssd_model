import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np

import base64
import io

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing


# POSTPROCESSING PARAMS

class SSDServer:
    NET_SHAPE = (300, 300)
    SELECT_TRESHOLD = 0.5

    NMS_TRESHOLD = 0.45
    TOP_K = 400

    ssd_net = ssd_vgg_300.SSDNet()
    ssd_anchors = ssd_net.anchors(NET_SHAPE)

    VOC_MAP = {
        0: 'none',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    def __init__(self, session, inputs, outputs):
        self.session = session
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def initialize(model_path: str, signature_name: str):
        session = tf.Session()
        meta_graph = tf.saved_model.loader.load(session, [tf.saved_model.tag_constants.SERVING], model_path)

        signature = meta_graph.signature_def[signature_name]
        inputs = {}
        for inp_name, inp in signature.inputs.items():
            inputs[inp_name] = session.graph.get_tensor_by_name(inp.name)

        outputs = {}
        for out_name, out in signature.outputs.items():
            outputs[out_name] = session.graph.get_tensor_by_name(out.name)

        return SSDServer(session, inputs, outputs)

    def preprocess(self, b64_string: str):
        decoded = base64.b64decode(b64_string)
        jpeg_image = io.BytesIO(decoded)
        return mpimg.imread(jpeg_image, format='JPG')

    def run(self, pic_matrix):
        return self.session.run(self.outputs, feed_dict={self.inputs["img_input"]: pic_matrix})

    def postprocess(self, rpredictions, rlocalisations, rbbox_img):
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, SSDServer.ssd_anchors,
            select_threshold=SSDServer.SELECT_TRESHOLD, img_shape=SSDServer.NET_SHAPE, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)

        rclasses, rscores, rbboxes = np_methods.bboxes_sort(
            rclasses, rscores, rbboxes,
            top_k=SSDServer.TOP_K
        )

        rclasses, rscores, rbboxes = np_methods.bboxes_nms(
            rclasses, rscores, rbboxes,
            nms_threshold=SSDServer.NMS_TRESHOLD

        )
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)

        rclasses = list(map(lambda c: SSDServer.VOC_MAP.get(c, "NA"), rclasses))
        return rclasses, rscores, rbboxes
