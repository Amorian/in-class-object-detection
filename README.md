# In Class Object Detection with Real Time Inference on Kubernetes Deployment

Used ~20 images from class to build a simple object detector with real time inference on a flask web service deployment through Kubernetes

Actual images excluded due to privacy reasons

## Training

* Use the `environment.yml` file to set up using Anaconda
* Annotate your images with any tool of your choice that outputs to the PASCAL-VOC XML format
* Run the code and use the model that performs well.

## Inference

* Use `image-recognition.yaml` to deploy on Kubernetes
* Create a service to make it accessible
* Use the server endpoint to access the Flask application
