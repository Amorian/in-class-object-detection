import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageFile

import os
import cv2
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template

PORT = 8080

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["IMAGE_UPLOADS"] = 'static/images/'
ImageFile.LOAD_TRUNCATED_IMAGES = True
VOC_BBOX_LABEL_NAMES = ('__background__', 'laptop', 'empty chair')
num_classes = 3
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('model7.pth', map_location=torch.device('cpu')))
model.eval()
os.remove('/root/.cache/torch/checkpoints/resnet50-19c8e357.pth')
os.remove('model7.pth')

def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    os.remove(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_class = [VOC_BBOX_LABEL_NAMES[i] for i in list(pred[0]['labels'].detach().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class

def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    img = cv2.imread(img_path) # Read image with cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
    boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) # Write the prediction class
    plt.figure(figsize=(20,30)) # display the output image
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('static/images/plot.png')
    plt.close()

@app.route("/")
def hello():
    return "Faster R-CNN on pictures of the class to identify laptops and empty chairs"

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        img_path = os.path.join(app.config["IMAGE_UPLOADS"], f.filename)
        f.save(img_path)
        object_detection_api(img_path, threshold=0.8)
        return render_template('result.html', url='static/images/plot.png')
    return render_template('upload-image.html')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
