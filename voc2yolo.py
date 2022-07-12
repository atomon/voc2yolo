# Convert VOC format(labelme) to Yolo Format
# python voc2yolo.py --images ../datasets/JPEGImages --annotations ../datasets/Annotations --classes ../datasets/class_names.txt --outputs ../datasets/yolo_datasets --ratio 0.8


import argparse
import random
import shutil
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from ruamel.yaml import YAML

def make_parser():
    parser = argparse.ArgumentParser('YOLOX_COCO_dataset_generator')

    parser.add_argument("--images", type=str, default=None, help="path to VOC dataset images")
    parser.add_argument("--annotations", type=str, default=None, help="path to VOC dataset annotations")
    parser.add_argument("--classes", type=str, default=None, help="class (*.txt path)")
    parser.add_argument("--outputs", type=str, default='./yolo_datasets', help="save datasets path")
    parser.add_argument("--ratio", type=float, default=0.8, help="train/validation split ratio(train)")

    return parser

def createLabel(ann_xml, ann_txt, classes):
    tree = ET.parse(ann_xml)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with (ann_txt / annotations_list[index].with_suffix('.txt').name).open(mode='a') as f:
        for obj in root.findall('object'):
            cls = obj.find('name').text
            if cls not in classes:
                continue
            cls_id = classes.index(cls)

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            xCentor = ((xmin + xmax) / 2.0 -1) / w
            yCentor = ((ymin + ymax) / 2.0 -1) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            f.write("%d %.06f %.06f %.06f %.06f\n" % (cls_id, xCentor, yCentor, width, height))


if __name__ == "__main__":
    args = make_parser().parse_args()

    # make directries
    outputs = Path(args.outputs)
    (outputs / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (outputs / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (outputs / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (outputs / 'val' / 'labels').mkdir(parents=True, exist_ok=True)

    # read classes
    classes_file = Path(args.classes)
    with classes_file.open(mode='r') as f:
        classes = [s.strip() for s in f.readlines()]
        classes.remove('_background_')

    # make coco.yaml
    train_images = args.outputs + '/train/images'
    val_images = args.outputs + '/val/images'
    coco_dict = {'train': train_images, 
                 'val': val_images,
                 'nc': len(classes),
                 'names': classes}
    coco_yaml = outputs / 'coco.yaml'
    yaml = YAML()
    yaml.default_flow_style = None 
    yaml.dump(coco_dict, coco_yaml)


    # Percentage of training data
    train_ratio = args.ratio
    
    # Get and sort data file list
    images_dir = Path(args.images)
    annotations_dir = Path(args.annotations)
    annotations_list = sorted(annotations_dir.glob('*.xml'))
    file_num = len(annotations_list)
    
    # index shuffle
    index_list = list(range(file_num - 1))
    random.shuffle(index_list)
    
    for count, index in enumerate(index_list):
        if count < int(file_num * train_ratio):
            # Training Data
            createLabel(annotations_list[index], outputs / 'train' / 'labels', classes)
            shutil.copy2(images_dir / annotations_list[index].with_suffix('.jpg').name, 
                         outputs / 'train' / 'images')
        else:
            # Validation Data
            createLabel(annotations_list[index], outputs / 'val' / 'labels', classes)
            shutil.copy2(images_dir / annotations_list[index].with_suffix('.jpg').name, 
                         outputs / 'val' / 'images')
