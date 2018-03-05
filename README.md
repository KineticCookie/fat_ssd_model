# hs_ssd_model
SSD object detection model.

Written as an example for Hydroserving platform, and particularly for `hs` CLI tool.

## Serving metadata
In addition to SSD model, this repo contains following files:
- [serving.yaml](/serving.yaml)
- [contract.prototxt](/contract.prototxt)

`serving.yaml` contains description of a model and parameters for local deploy, which can be used for development purposes.
`contract.prototxt` contains ModelContract protobuf message which describes contract of this model.


## How to deploy the model locally
1. Pack the model  
`hs pack`
2. Start local runtime  
`hs local start`

Now, there is a docker container with this model which you can use.

## How to upload the model to Hydroserving instance
1. Assemble the model  
`hs assemble`
2. Upload the model  
`hs upload --host {HYDROSERVING_HOST} --port {HYDROSERVING_PORT} --source {SOME_MODEL_SOURCE}`
