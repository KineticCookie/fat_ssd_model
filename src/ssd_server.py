import tensorflow as tf
import matplotlib.image as mpimg

import base64
import io

from nets import ssd_vgg_300
from nets import np_methods

import hydro_serving_grpc as hs


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
    def initialize(model_path, signature_name):
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

    def preprocess(self, b64_string):
        decoded = base64.b64decode(b64_string)
        jpeg_image = io.BytesIO(decoded)
        return mpimg.imread(jpeg_image, format='JPG')

    def run(self, pic_matrix):
        outputs = self.session.run(self.outputs, feed_dict={self.inputs["img_input"]: pic_matrix})
        rpredictions = [
            outputs['ssd_300_vgg/softmax/Reshape_1:0'],
            outputs['ssd_300_vgg/softmax_1/Reshape_1:0'],
            outputs['ssd_300_vgg/softmax_2/Reshape_1:0'],
            outputs['ssd_300_vgg/softmax_3/Reshape_1:0'],
            outputs['ssd_300_vgg/softmax_4/Reshape_1:0'],
            outputs['ssd_300_vgg/softmax_5/Reshape_1:0']
        ]
        rlocalisations = [
            outputs['ssd_300_vgg/block4_box/Reshape:0'],
            outputs['ssd_300_vgg/block7_box/Reshape:0'],
            outputs['ssd_300_vgg/block8_box/Reshape:0'],
            outputs['ssd_300_vgg/block9_box/Reshape:0'],
            outputs['ssd_300_vgg/block10_box/Reshape:0'],
            outputs['ssd_300_vgg/block11_box/Reshape:0']
        ]
        rbbox_img = outputs['bbox_img']

        return rpredictions, rlocalisations, rbbox_img

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

        class_arr = [bytes(x, "utf-8") for x in rclasses]
        classes_tensor = hs.TensorProto(
            dtype=hs.DT_STRING,
            tensor_shape=hs.TensorShapeProto(
                dim=[
                    hs.TensorShapeProto.Dim(size=-1)
                ]
            ),
            string_val=class_arr
        )

        scores_tensor = hs.TensorProto(
            dtype=hs.DT_DOUBLE,
            tensor_shape=hs.TensorShapeProto(
                dim=[
                    hs.TensorShapeProto.Dim(size=-1)
                ]
            ),
            double_val=rscores
        )

        bboxes_tensor = hs.TensorProto(
            dtype=hs.DT_DOUBLE,
            tensor_shape=hs.TensorShapeProto(
                dim=[
                    hs.TensorShapeProto.Dim(size=-1),
                    hs.TensorShapeProto.Dim(size=4)
                ]
            ),
            double_val=rbboxes.flatten()
        )

        return classes_tensor, scores_tensor, bboxes_tensor
