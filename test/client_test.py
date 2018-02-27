import unittest
import base64
import hydro_serving_grpc as hs
import grpc


class ClientTest(unittest.TestCase):
    def test_runtime(self):
        channel = grpc.insecure_channel('localhost:6969')
        client = hs.PredictionServiceStub(channel=channel)

        with open("test_pic.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        image_tensor = hs.TensorProto(
            dtype=hs.DT_STRING,
            string_val=[encoded_string]
        )

        request = hs.PredictRequest(
            model_spec=hs.ModelSpec(signature_name="detect"),
            inputs={
                "image_b64": image_tensor,
            }
        )

        result = client.Predict(request)
        print(result)


if __name__ == "__main__":
    unittest.main()
