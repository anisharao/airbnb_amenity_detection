import os
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
from collections import Counter

import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw
import requests
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import pymysql

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
    print('json_file',json_file)
    # Check to see if json_file exists
    assert json_file, "No .json label file found, please make one with get_image_dicts()"

    with open(json_file, "r") as f:
        img_dicts = json.load(f)

    # Convert bbox_mode to Enum of BoxMode.XYXY_ABS (doesn't work loading normal from JSON)
    for img_dict in img_dicts:
        for annot in img_dict["annotations"]:
            annot["bbox_mode"] = BoxMode.XYXY_ABS

    return img_dicts


def load_pretrained_fasterrcnn_model(uploaded_image):
    # Load the pretrained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    # image = Image.open(uploaded_image)

    coco_classes = [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
        "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Perform object detection
    with torch.no_grad():
        predictions = model([image_tensor])

    # Get the boxes, labels, and scores from the predictions
    boxes = predictions[0]['boxes']
    labels = predictions[0]['labels']
    scores = predictions[0]['scores']

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Iterate through the detected objects and filter by class name
    for score, label, box in zip(scores, labels, boxes):
        # Get the class name corresponding to the label
        # print("label",label)
        if label.item() > 80 or score < .60:
            continue;
        class_name = coco_classes[label.item()]
        print(coco_classes)
        # Check if the detected class name is in the list of classes to display
        if class_name in coco_classes:
            print(f"Displaying: Class {label.item()}, Score {score:.2f}")
            # Convert box coordinates to integers
            box = [int(coord) for coord in box]

            # Draw the bounding box on the image
            draw.rectangle(box, outline="black", width=2)

            # Add class name and score as text
            label_text = f"{class_name} ({score:.2f})"
            draw.text((box[0], box[1]), label_text, fill="red")

    # image.show()  # To display the image with bounding boxes and scores
    # image.save("output_image.jpg")  # To save the image with bounding boxes and scores
    return image

def register():
    # Register datasets with Detectron2
    print(f"Registering airbnb-openImagesV7/train")
    if 'airbnb-openImagesV7/train' not in DatasetCatalog.list():
        DatasetCatalog.register('airbnb-openImagesV7/train', lambda dataset_name='airbnb-openImagesV7/train':load_json_labels('/content/drive/Shareddrives/Data 298A/merged_file.json'))
        MetadataCatalog.get('airbnb-openImagesV7/train').set(thing_classes=classes)

    print(f"Registering airbnb-openImagesV7/validation")
    if 'airbnb-openImagesV7/validation' not in DatasetCatalog.list():
        DatasetCatalog.register('airbnb-openImagesV7/validation', lambda dataset_name='airbnb-openImagesV7/validation':load_json_labels('/content/drive/Shareddrives/Data 298A/Data/fiftyone/open-images-v7/validation/validation_labels.json'))
        MetadataCatalog.get('airbnb-openImagesV7/validation').set(thing_classes=classes)
    return MetadataCatalog.get("airbnb-openImagesV7/train")



def retina_model():
    # Setup a model config (recipe for training a Detectron2 model)
    cfg = get_cfg()
    # Add some basic instructions for the Detectron2 model from the model_zoo: https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    # Add some pretrained model weights from an object detection model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_50_FPN_3x.yaml")
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
    # Get the final model weights from the outputs directory
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_50_3k.pth")
    # Set the testing threshold (a value between 0 and 1, higher makes it more difficult for a prediction to be made)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # Tell the config what the test dataset is (we've already done this)
    cfg.DATASETS.TEST = ("airbnb-openImagesV7/validation",)
    # Setup a default predictor from Detectron2: https://detectron2.readthedocs.io/modules/engine.html#detectron2.engine.defaults.DefaultPredictor
    custom_model_weights_path = '/content/drive/Shareddrives/Data 298A/retinanet_model_50_3k.pth'
    trainer.load_state_dict(torch.load(custom_model_weights_path))
    cfg.MODEL.WEIGHTS = os.path.join('/content/drive/Shareddrives/Data 298A/model_final_50_3k.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)
    return predictor

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

def upload_image(image,predictor,airbnb_openV7_metadata,airbnb_id):
        # test_image = Image.open(destination_path)
        confidence_threshold = 0.7
        image.save("uploaded_image.jpg")
        im = cv2.imread('uploaded_image.jpg')
        outputs = predictor(im)
        # Filter instances based on confidence score
        instances = outputs["instances"]
        above_threshold = instances[instances.scores > confidence_threshold]
        # Get detected labels
        detected_labels = above_threshold.pred_classes.tolist()

        # Create a list to store the corresponding label names
        label_names = []

        # Map label IDs to label names using the metadata
        for label_id in detected_labels:
            label_name = airbnb_openV7_metadata.thing_classes[label_id]
            label_names.append(label_name)
        # check_labels = insert_or_get_columns_with_value_1(file_name, label_names)
        # missing_amenities_list = list(set(check_labels) - set(label_names))
        # Count the occurrences of each label
        label_counts = Counter(label_names)
        # Create a string in the desired format
        detected_labels_str = ', '.join(f'{label}={count}' for label, count in label_counts.items())
        for label, count in label_counts.items():
            insert_into_airbnb_detection(airbnb_id,label, count)
        print('Inserted details in airbnb_detection successfully')
        v = Visualizer(im[:, :, ::-1], metadata=airbnb_openV7_metadata, scale=0.8)
        # Draw only the instances above the confidence threshold
        v = v.draw_instance_predictions(above_threshold.to("cpu"))
        plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
        return Image.fromarray(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)), detected_labels_str
        # if missing_amenities_list:
        #     # Create a string with the missing labels
        #     missing_labels = ', '.join(str(label) for label in missing_amenities_list)
        #     return Image.fromarray(
        #         cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)), detected_labels_str, missing_labels
        # else:
        #     return Image.fromarray(
        #         cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)), detected_labels_str, "No missing amenities"

def insert_into_airbnb_registration(airbnb_id, airbnb_name, airbnb_location, airbnb_price):
    connection = pymysql.connect(host="34.70.162.68", user="anisharao", passwd="airbnb", db="airbnb-298")
    try:
        cursor = connection.cursor()

        # Check if the row already exists
        check_query = "SELECT COUNT(*) FROM airbnb_registration WHERE airbnb_id = %s"
        cursor.execute(check_query, (airbnb_id,))
        row_count = cursor.fetchone()[0]

        if row_count == 0:
            # The row does not exist, proceed with the insertion
            insert_query = """
                INSERT INTO airbnb_registration (airbnb_id, airbnb_name, airbnb_location, airbnb_price)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (airbnb_id, airbnb_name, airbnb_location, airbnb_price))
            connection.commit()
            return True
        else:
            # The row already exists, do not insert
            print(f"Row with airbnb_id {airbnb_id} already exists.")
            return False

    except pymysql.Error as e:
        print(f"Error: {e}")
        return False

    finally:
        cursor.close()
        connection.close()

# Function to fetch amenity_id based on amenity_name
def fetch_amenity_id(amenity_name):
    connection = pymysql.connect(host="34.70.162.68", user="anisharao", passwd="airbnb", db="airbnb-298")
    try:
        cursor = connection.cursor()
        select_query = "SELECT amenity_id FROM amenities WHERE amenity_name = %s"
        cursor.execute(select_query, (amenity_name,))
        result = cursor.fetchone()

        if result:
            return result[0]
        else:
            return None

    finally:
        cursor.close()
        connection.close

# Function to insert data into the airbnb_detection table
def insert_into_airbnb_detection(airbnb_id, amenity_name,amenity_count):
    connection = pymysql.connect(host="34.70.162.68", user="anisharao", passwd="airbnb", db="airbnb-298")
    try:
        cursor = connection.cursor()

        # Fetch amenity_id based on amenity_name
        amenity_id = fetch_amenity_id(amenity_name)
        print('Amenity id found in database is :: ', amenity_id)

        if amenity_id is not None:
            # Example SQL query (modify according to your table structure)
            insert_query = """
                INSERT INTO airbnb_detection (airbnb_id, amenity_id,amenity_count)
                VALUES (%s, %s,%s)
            """
            cursor.execute(insert_query, (airbnb_id, amenity_id,amenity_count))
            connection.commit()
            return True
        else:
            return False  # Amenity not found

    except pymysql.Error as e:
        print(f"Error: {e}")
        return False

    finally:
        cursor.close()
        connection.close