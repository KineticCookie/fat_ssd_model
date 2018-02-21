import hydro_serving_grpc as hs
from ssd_server import SSDServer

# INITIALIZE

ssd = SSDServer.initialize("model/", "serving_default")


# MODEL ENTRYPOINT

def detect(image_b64):
    pic_matrix = ssd.preprocess(image_b64.str_val[0])
    rpredictions, rlocalisations, rbbox_img = ssd.run(pic_matrix)
    rclasses, rscores, rbboxes = ssd.postprocess(rpredictions, rlocalisations, rbbox_img)
    return hs.PredictResponse(
        outputs={
            "classes": rclasses,
            "scores": rscores,
            "bboxes": rbboxes
        }
    )
