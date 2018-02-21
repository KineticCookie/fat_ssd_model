import unittest
from ssd_server import SSDServer
import base64
import hydro_serving_grpc as hs
from google.protobuf import text_format


class SSDSpecs(unittest.TestCase):

    def test_contract(self):
        with open("serving_metadata/contract.prototxt", "r") as contract_file:
            contract = hs.ModelContract()
            text_format.Merge(contract_file.read(), contract)
        print(contract)

    def test_model(self):
        ssd = SSDServer.initialize("model", "serving_default")
        with open("test_pic.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        pix_mat = ssd.preprocess(encoded_string)

        print(ssd.inputs)
        print(ssd.outputs)
        outputs = ssd.run(pix_mat)

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

        rclasses, rscores, rbboxes = ssd.postprocess(rpredictions, rlocalisations, rbbox_img)
        print(rclasses)
        print(rscores)
        print(rbboxes)

        result_classes = [x.decode("utf-8") for x in rclasses.string_val]

        self.assertEqual(rclasses.dtype, hs.DT_STRING)
        self.assertSetEqual(set(result_classes), {"dog", "person"})


if __name__ == "__main__":
    unittest.main()
