import hydro_serving_grpc as hs
import ssd_server

# INITIALIZE

session, outputs = ssd_server.initialize()


# MODEL ENTRYPOINT

def detect(image_b64):
    pic_matrix = ssd_server.preprocess(image_b64.str_val[0])
    rpredictions, rlocalisations, rbbox_img = ssd_server.run(session, outputs, pic_matrix)
    rclasses, rscores, rbboxes = ssd_server.postprocess(rpredictions, rlocalisations, rbbox_img)
    return hs.PredictResponse(
        outputs={
            "classes": rclasses,
            "scores": rscores,
            "bboxes": rbboxes
        }
    )
