import argparse, json
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from statistics import mean




def calculate_f1_score(gt_mask, pred_mask):
  pred_list = []
  for i in range(0,len(pred_mask),2):
    pred_list.append([pred_mask[i],pred_mask[i+1]])
  gt_list = []
  for i in range(0,len(gt_mask),2):
    gt_list.append([gt_mask[i], gt_mask[i+1]])

  intersection = Polygon(pred_list).intersection(Polygon(gt_list))
  if str(type(intersection)) != "<class 'shapely.geometry.polygon.Polygon'>":
    intersection = MultiPolygon([x for x in intersection.geoms if str(type(x)) == "<class 'shapely.geometry.polygon.Polygon'>"])

  tp = intersection.area
  # print(tp)
  # print(Polygon(pred_list).intersection(Polygon(gt_list)))
  
  if tp == 0:
    return 0
  fp = (Polygon(pred_list) - intersection).area
  fn = (Polygon(gt_list) - intersection).area
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  f1_score = 2*(precision * recall)/ (precision + recall)
  return f1_score

def matching_annotations(gt_annotation, pred_annotation):
  #assuming gt and pred category ids are equal - will be provided
  assert len(gt_annotation['images'])==len(pred_annotation['images']) and len(gt_annotation['categories'])==len(pred_annotation['categories']) 
  num_images = len(gt_annotation['images'])
  list_images_information = []
  for i in gt_annotation['images']:
    temp_dict = {}
    temp_dict['image_id'] = i['id']
    temp_dict['image_name'] = i['file_name']
    list_images_information.append(temp_dict)
  file_name_dict_pred = {}
  for i in pred_annotation['images']:
    file_name_dict_pred[i['id']] = i['file_name']

  file_name_dict_gt = {}
  for i in gt_annotation['images']:
    file_name_dict_gt[i['id']] = i['file_name']

  for m in range(0, len(pred_annotation['annotations'])):
    pred_annotation['annotations'][m]['file_name'] = file_name_dict_pred[pred_annotation['annotations'][m]['image_id']]

  for m in range(0, len(gt_annotation['annotations'])):
    gt_annotation['annotations'][m]['file_name'] = file_name_dict_gt[gt_annotation['annotations'][m]['image_id']]

  for i in range(0, len(list_images_information)):
    temp_list_classes_gt=[]
    temp_list_classes_pred=[]
    for z in gt_annotation['annotations']:
      if z['image_id']==list_images_information[i]['image_id']:
        temp_list_classes_gt.append(z['category_id'])
    for k in pred_annotation['annotations']:
      if k['file_name']==list_images_information[i]['image_name']:
        temp_list_classes_pred.append(k['category_id'])
    list_images_information[i]['gt_classes'] = sorted(temp_list_classes_gt)
    list_images_information[i]['pred_classes'] = sorted(temp_list_classes_pred)
  return gt_annotation, pred_annotation, list_images_information

def mean_f1_score(gt_annotation, pred_annotation, list_images_information):
  for i in range(0, len(list_images_information)):
    temp_dict_class_f1_score = {}
    for k in gt_annotation['annotations']:
      if list_images_information[i]['image_name']==k['file_name'] and k['category_id'] in list_images_information[i]['gt_classes']:
        for z in pred_annotation['annotations']:
          if k['file_name'] == z['file_name']:
            if z['category_id'] in list_images_information[i]['gt_classes']:
              if z['category_id'] == k['category_id']:
                if k['category_id'] in list(temp_dict_class_f1_score.keys()):
                  continue
                class_f1_score = calculate_f1_score(k['segmentation'][0], z['segmentation'][0])
                temp_dict_class_f1_score[k['category_id']] = class_f1_score
            else:
              temp_dict_class_f1_score[k['category_id']] = 0
    for gt_cat in list_images_information[i]['gt_classes']:
      if gt_cat not in list_images_information[i]['pred_classes']:
        temp_dict_class_f1_score[gt_cat] = 0
    list_images_information[i]['f1_scores']=temp_dict_class_f1_score
  all_means = []
  for i in range(0, len(list_images_information)):
    list_of_f1_scores_of_image = list(list_images_information[i]['f1_scores'].values())
    list_images_information[i]['mean_f1_score'] = mean(list_of_f1_scores_of_image)
    all_means.append(list_images_information[i]['mean_f1_score'])
  
  return mean(all_means), all_means, list_images_information
  
def matching_annotations_stenosis(gt_annotation, pred_annotation):
  #assuming gt and pred category ids are equal - will be provided
  assert len(gt_annotation['images'])==len(pred_annotation['images']) and len(gt_annotation['categories'])==len(pred_annotation['categories']) 
  num_images = len(gt_annotation['images'])
  list_images_information = []
  for i in gt_annotation['images']:
    temp_dict = {}
    temp_dict['image_id'] = i['id']
    temp_dict['image_name'] = i['file_name']
    list_images_information.append(temp_dict)
  file_name_dict_pred = {}
  for i in pred_annotation['images']:
    file_name_dict_pred[i['id']] = i['file_name']

  file_name_dict_gt = {}
  for i in gt_annotation['images']:
    file_name_dict_gt[i['id']] = i['file_name']

  for m in range(0, len(pred_annotation['annotations'])):
    pred_annotation['annotations'][m]['file_name'] = file_name_dict_pred[pred_annotation['annotations'][m]['image_id']]

  for m in range(0, len(gt_annotation['annotations'])):
    gt_annotation['annotations'][m]['file_name'] = file_name_dict_gt[gt_annotation['annotations'][m]['image_id']]

  for i in range(0, len(list_images_information)):
    temp_list_classes_gt=[]
    temp_list_classes_pred=[]
    for z in gt_annotation['annotations']:
      if z['image_id']==list_images_information[i]['image_id']:
        temp_list_classes_gt.append(z['category_id'])
    for k in pred_annotation['annotations']:
      if k['file_name']==list_images_information[i]['image_name']:
        temp_list_classes_pred.append(k['category_id'])
    list_images_information[i]['num_gt_masks'] = len(temp_list_classes_gt)
    list_images_information[i]['num_pred_masks'] = len(temp_list_classes_pred)
  return gt_annotation, pred_annotation, list_images_information


def mean_f1_score_stenosis(gt_annotation, pred_annotation, list_images_information):
  for i in range(0, len(list_images_information)):
    temp_list_class_f1_score = []
    for k in pred_annotation['annotations']:
      if list_images_information[i]['image_name']==k['file_name']:
        for z in gt_annotation['annotations']:
          if k['file_name'] == z['file_name']:
              class_f1_score = calculate_f1_score(k['segmentation'][0], z['segmentation'][0])
              if class_f1_score>0:
                temp_list_class_f1_score.append(class_f1_score)
                break
              else: continue
    for j in range(0, list_images_information[i]['num_gt_masks']-len(temp_list_class_f1_score)):
      temp_list_class_f1_score.append(0)
    
    list_images_information[i]['f1_scores']=temp_list_class_f1_score
  all_means = []
  for i in range(0, len(list_images_information)):
    list_images_information[i]['mean_f1_score'] = mean(list_images_information[i]['f1_scores'])
    all_means.append(list_images_information[i]['mean_f1_score'])
  
  return mean(all_means), all_means, list_images_information



def main():
  parser = argparse.ArgumentParser('Compute F1 score', add_help=False)
  parser.add_argument('--gt_json', type=str, help='name of ground truth annotation file in COCO format', required=True)
  parser.add_argument('--pred_json', type=str, help='name of predicted annotation file in COCO format', required=True)
  parser.add_argument('--stenosis', default='False', help='name of file with results', required=False)
  parser.add_argument('--results_name', default='results', help='name of file with results')
  args = parser.parse_args()

  gt_annotation = args.gt_json
  pred_annotation = args.pred_json
  pred_annotation = open(pred_annotation)
  pred_annotation = json.load(pred_annotation)
  gt_annotation = open(gt_annotation)
  gt_annotation = json.load(gt_annotation)
  if args.stenosis == False:
    gt_annotation, pred_annotation, list_images_information = matching_annotations(gt_annotation, pred_annotation)
    total_mean, list_of_all_means,total_info = mean_f1_score(gt_annotation, pred_annotation, list_images_information)
  else:
    gt_annotation, pred_annotation, list_images_information = matching_annotations_stenosis(gt_annotation, pred_annotation)
    total_mean, list_of_all_means,total_info = mean_f1_score_stenosis(gt_annotation, pred_annotation, list_images_information)
  whole_info = {}
  whole_info['total mean'] = total_mean
  whole_info['all means per image'] = list_of_all_means
  whole_info['total info'] = total_info
  with open(f'{args.results_name}.json', 'w', encoding='ascii') as f:
            json.dump(whole_info,f)
  return whole_info

if __name__ == '__main__':
    main()