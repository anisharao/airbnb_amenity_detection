from detectron2.structures import BoxMode
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
import pymysql



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
def insert_into_airbnb_detection(airbnb_id, amenity_name, amenity_count, category):
    connection = pymysql.connect(host="34.70.162.68", user="anisharao", passwd="airbnb", db="airbnb-298")
    try:
        cursor = connection.cursor()

        # Check if the row already exists
        check_query = "SELECT amenity_count FROM airbnb_detection WHERE airbnb_id = %s and amenity_name = %s"
        cursor.execute(check_query, (airbnb_id, amenity_name))
        existing_count = cursor.fetchone()

        if existing_count is not None:
            # If the row exists, update it
            update_query = """
                UPDATE airbnb_detection
                SET amenity_count = %s
                WHERE airbnb_id = %s AND amenity_name = %s
            """
            cursor.execute(update_query, (amenity_count, airbnb_id, amenity_name))
        else:
            # If the row doesn't exist, insert a new row
            insert_query = """
                INSERT INTO airbnb_detection (airbnb_id, amenity_name, amenity_count, category)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(insert_query, (airbnb_id, amenity_name, amenity_count, category))

        connection.commit()
        return True

    except pymysql.Error as e:
        print(f"Error: {e}")
        return False

    finally:
        cursor.close()
        connection.close()
