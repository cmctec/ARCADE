# Useful scripts page

### COCO To YOLO

To convert COCO dataset, provided with the challenge, to YOLOv8 annotations format, simply paste the json file path into '''coco_path''' variable and run the notebook. To use this dataset with YOLOv8 it is important to also include the '''data.yaml''' file, which contains the information about class labels and train/val/test splits.

### Create Masks

To create segmentation masks, suitable for U-Net models, from COCO Json annotations, you need to paste the json file path into '''coco_path''' variable and run the notebook. '''gt_mask''' variable will contain a matrix with class masks in format '''[image, class, x, y]'''.

### Mask To COCO

To convert the result of U-Net-like model, which outputs masks as a matrix like '''[image, class, x, y]''', you should change the '''gt_mask''' variable to your desired output matrix, and run the notebook. This way you will get '''submit.json''', which you can submit directly to the Challenge webpage. Notice: '''empty_annotations.json''' file contains pre-made template, to which you only need to fill the segmentation results.

### YOLO to COCO

To convert your YOLO labels back to COCO annotations format, you should first consult your '''data.yaml''' file, and compare the ids of categories in it with the category ids in '''empty_annotations.json''', and create a corresponding mapping in a dictionary. You then need to specify the '''YOLO_LABELS_FOLDER''', and '''SAVE_PATH''', and run the notebook to produce the '''annotations.json''' file with your results