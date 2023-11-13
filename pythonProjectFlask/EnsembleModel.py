import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import Utility
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import BoxMode, Instances
import json
import cv2
from collections import Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes

from detectron2.utils.visualizer import Visualizer

# Target classes with spaces removed
classes = ['Bathtub',
           'Bed',
           'Billiard table',
           'Ceiling fan',
           'Coffeemaker',
           'Couch',
           'Countertop',
           'Dishwasher',
           'Fireplace',
           'Fountain',
           'Gas stove',
           'Jacuzzi',
           'Kitchen & dining room table',
           'Microwave oven',
           'Mirror',
           'Oven',
           'Pillow',
           'Porch',
           'Refrigerator',
           'Shower',
           'Sink',
           'Sofa bed',
           'Stairs',
           'Swimming pool',
           'Television',
           'Toilet',
           'Towel',
           'Tree house',
           'Washing machine',
           'Wine rack']


def load_json_labels(json_file):
    print('json_file', json_file)
    # Check to see if json_file exists
    assert json_file, "No .json label file found, please make one with get_image_dicts()"

    with open(json_file, "r") as f:
        img_dicts = json.load(f)

    # Convert bbox_mode to Enum of BoxMode.XYXY_ABS (doesn't work loading normal from JSON)
    for img_dict in img_dicts:
        for annot in img_dict["annotations"]:
            annot["bbox_mode"] = BoxMode.XYXY_ABS

    return img_dicts


def filter_classes(outputs, desired_class_ids, oim, confidence_threshold=0.61):
    cls = outputs['instances'].pred_classes
    scores = outputs["instances"].scores
    boxes = outputs['instances'].pred_boxes

    # Index to keep where class is in the desired class IDs
    indices_to_keep = [i for i, cls_id in enumerate(cls) if cls_id in desired_class_ids]

    # Only keep predictions for the desired classes
    filtered_cls = torch.tensor(np.take(cls.cpu().numpy(), indices_to_keep))
    filtered_scores = torch.tensor(np.take(scores.cpu().numpy(), indices_to_keep))
    filtered_boxes = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indices_to_keep, axis=0)))

    # Further filter based on confidence scores
    high_confidence_indices = [i for i, score in enumerate(filtered_scores) if score >= confidence_threshold]

    filtered_cls = torch.tensor(np.take(filtered_cls.cpu().numpy(), high_confidence_indices))
    filtered_scores = torch.tensor(np.take(filtered_scores.cpu().numpy(), high_confidence_indices))
    filtered_boxes = Boxes(torch.tensor(np.take(filtered_boxes.tensor.cpu().numpy(), high_confidence_indices, axis=0)))

    # Create a new Instances object and set its fields
    obj = Instances(image_size=(oim.shape[0], oim.shape[1]))
    obj.set('pred_classes', filtered_cls)
    obj.set('scores', filtered_scores)
    obj.set('pred_boxes', filtered_boxes)

    return obj


def filter_classes_original(outputs, oim, confidence_threshold=0.61):
    cls = outputs['instances'].pred_classes
    scores = outputs["instances"].scores
    boxes = outputs['instances'].pred_boxes

    # Further filter based on confidence scores
    high_confidence_indices = [i for i, score in enumerate(scores) if score >= confidence_threshold]

    filtered_cls = torch.tensor(np.take(cls.cpu().numpy(), high_confidence_indices))
    filtered_scores = torch.tensor(np.take(scores.cpu().numpy(), high_confidence_indices))
    filtered_boxes = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), high_confidence_indices, axis=0)))

    # Create a new Instances object and set its fields
    obj = Instances(image_size=(oim.shape[0], oim.shape[1]))
    obj.set('pred_classes', filtered_cls)
    obj.set('scores', filtered_scores)
    obj.set('pred_boxes', filtered_boxes)

    return obj


def load_pretrained_maskRCNN_Model():
    # Load a pretrained Mask R-CNN model
    mask_cfg = get_cfg()
    mask_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    mask_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    mask_predictor = DefaultPredictor(mask_cfg)
    return mask_cfg, mask_predictor


def fasterrcnn_model():
    # Setup a model config (recipe for training a Detectron2 model)
    cfg = get_cfg()
    # Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # Add some pretrained model weights from an object detection model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    # Setup datasets to train/validate on (this will only work if the datasets are registered with DatasetCatalog)
    cfg.DATASETS.TRAIN = ("airbnb-openImagesV7/train",)
    cfg.DATASETS.TEST = ("airbnb-openImagesV7/validation",)
    # How many dataloaders to use? This is the number of CPUs to load the data into Detectron2, Colab has 2, so we'll use 2
    cfg.DATALOADER.NUM_WORKERS = 2
    # How many images per batch? The original models were trained on 8 GPUs with 16 images per batch, since we have 1 GPU: 16/8 = 2.
    cfg.SOLVER.IMS_PER_BATCH = 2
    # We do the same calculation with the learning rate as the GPUs, the original model used 0.01, so we'll divide by 8: 0.01/8 = 0.00125.
    cfg.SOLVER.BASE_LR = 0.00125
    # How many iterations are we going for?
    cfg.SOLVER.MAX_ITER = 3000
    # ROI = region of interest, as in, how many parts of an image are interesting, how many of these are we going to find?
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.RETINANET.NUM_CLASSES = 30
    # Setup output directory, all the model artefacts will get stored here in a folder called "outputs"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Setup the default Detectron2 trainer, see: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
    trainer = DefaultTrainer(cfg)
    # Set the testing threshold (a value between 0 and 1, higher makes it more difficult for a prediction to be made)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # Tell the config what the test dataset is (we've already done this)
    cfg.DATASETS.TEST = ("airbnb-openImagesV7/validation",)
    # Setup a default predictor from Detectron2: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultPredictor
    custom_model_weights_path = '/content/drive/Shareddrives/Data 298A/ver1/faster_model_50_3k_final.pth'
    trainer.load_state_dict(torch.load(custom_model_weights_path))
    cfg.MODEL.WEIGHTS = os.path.join('/content/drive/Shareddrives/Data 298A/ver1/model_final-fasterrcnn_4.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor


def process_uploaded_image(uploaded_image, airbnb_openV7_metadata, airbnb_id, category):
    oim = cv2.imread(uploaded_image)
    # Define the desired class IDs (0, 1, 2, 3, 4, etc.) for the classes you want to detect
    desired_class_ids_mask = [10, 13, 24, 25, 26, 28, 32, 33, 34, 39, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                              51, 52, 53, 54, 55, 56, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                              75, 76, 77, 78, 79]  # Modify this list according to your desired classes
    desired_class_ids_faster = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 19, 21, 22, 23, 26, 27, 28, 29]
    # Predictions and display only the specified classes
    mask_cfg, mask_predictor = load_pretrained_maskRCNN_Model()
    faster_predictor = fasterrcnn_model()
    original_outputs1 = filter_classes_original(mask_predictor(oim), oim)
    modified_outputs1 = filter_classes(mask_predictor(oim), desired_class_ids_mask, oim)
    original_outputs2 = filter_classes_original(faster_predictor(oim), oim)
    modified_outputs2 = filter_classes(faster_predictor(oim), desired_class_ids_faster, oim)

    mask_metadata = MetadataCatalog.get(mask_predictor.cfg.DATASETS.TRAIN[0])
    mask_label_names = [mask_metadata.thing_classes[label_id] for label_id in modified_outputs1.get('pred_classes')]

    faster_label_names = [airbnb_openV7_metadata.thing_classes[label_id] for label_id in
                          modified_outputs2.get('pred_classes')]

    instances1 = modified_outputs1
    num_instances1 = len(instances1)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    if num_instances1 > 0:
        # Loop through all instances
        for i in range(num_instances1):
            bbox = instances1.pred_boxes.tensor.numpy()[i]
            confidence = instances1.scores.numpy()[i]
            class_id = instances1.pred_classes.numpy()[i]
            class_label = coco_classes[class_id]
            # Create a Rectangle patch
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=1, edgecolor='b', facecolor='none')

            # Display the image with bounding box, confidence, and class label
            plt.imshow(oim[:, :, ::-1])  # Convert BGR to RGB
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1] - 5, f"Class: {class_label}\nConfidence: {confidence:.2f}",
                     color='blue', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    instances2 = modified_outputs2
    num_instances2 = len(instances2)

    if num_instances2 > 0:
        # Loop through all instances
        for i in range(num_instances2):
            bbox = instances2.pred_boxes.tensor.numpy()[i]
            confidence = instances2.scores.numpy()[i]
            class_id = instances2.pred_classes.numpy()[i]
            class_label = classes[class_id]
            # Create a Rectangle patch
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                     linewidth=1, edgecolor='b', facecolor='none')

            # Display the image with bounding box, confidence, and class label
            plt.imshow(oim[:, :, ::-1])  # Convert BGR to RGB
            plt.gca().add_patch(rect)
            plt.text(bbox[0], bbox[1] - 5, f"Class: {class_label}\nConfidence: {confidence:.2f}",
                     color='blue', fontsize=10, ha='left', va='bottom', bbox=dict(facecolor='white', alpha=0.8))

    plt.show()

    # Create a list containing label names
    label_names = mask_label_names + faster_label_names
    label_counts = Counter(label_names)
    detected_labels_str = ', '.join(f'{label}={count}' for label, count in label_counts.items())
    for label, count in label_counts.items():
        Utility.insert_into_airbnb_detection(airbnb_id, label, count, category)
    print(detected_labels_str)

    # Capture the current figure as an image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    # Convert the image to a NumPy array
    buf = io.BytesIO(buf.read())
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # pil_image = Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB))
    return pil_image, detected_labels_str


