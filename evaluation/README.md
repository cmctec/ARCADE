# Welcome to Evaluation Page



## Usage Example of evaluation_f1_score.py file:

  * To evaluate f1 score for segmentation, please run:
        '''
        python evaluation_f1_score.py --gt_json <ground_truth_annotation> --pred_json <predicted_annotation> --results_name <saved_results> --stenosis False
        '''
  * To evaluate f1 score for stenosis, please run:
        '''
        python evaluation_f1_score.py --gt_json <ground_truth_annotation> --pred_json <predicted_annotation> --results_name <saved_results> --stenosis True
        '''
        
### The results of evaluation_f1_score.py file is saved in json format with following structure:
  '''json
  {"total_mean": "total mean f1 score for whole image set (float)",
  "all means per image": "list of all means per image (list of floats)",
  "total info": {
                 "image_id": "image_id(int)",
                 "image_name": "file_name",
                 "gt_classes": "list of all ground truth classes that need to be present in image (for segmentation) ",
                 "pred_classes": "list of all predicted classes that are present in image (for segmentation) ",
                 "num_gt_masks":"number of ground truth masks that need to be present in image (for stenosis)",
                 "num_pred_masks":"number of predicted masks that are present in image (for stenosis)",
                 "f1_scores": "dictionary of classes and corresponding f1 score (for segmentation), list of f1 scores (for stenosis)",
                 "mean_f1_score": "mean f1 score for this image"
                  }
  }
  '''
        
** Note that all annotations are in specified COCO format, [HERE] ( https://cocodataset.org/#format-data) you can see the structure of COCO format for Object Detection. 
** Also note that category ids and their names are provided in **categories_segmentation.json** and **categories_stenosis.json** files


