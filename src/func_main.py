import hydro_serving_grpc as hs
from ssd_server import SSDServer
import os

# INITIALIZE
model_path = os.environ.get("MODEL_DIR")  # check if we are in runtime
if model_path is None:
    model_path = "../model"  # for local dev
else:
    model_path = os.path.join(model_path, "files", "model")

ssd = SSDServer.initialize(os.path.abspath(model_path), "serving_default")


# MODEL ENTRYPOINT

def detect(image_b64):
    pic_matrix = ssd.preprocess(image_b64.string_val[0].decode("utf-8"))
    rpredictions, rlocalisations, rbbox_img = ssd.run(pic_matrix)
    rclasses, rscores, rbboxes = ssd.postprocess(rpredictions, rlocalisations, rbbox_img)
    return hs.PredictResponse(
        outputs={
            "classes": rclasses,
            "scores": rscores,
            "bboxes": rbboxes
        }
    )
