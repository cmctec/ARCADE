from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from statistics import mean
from pathlib import Path
import json

class Arcade_phase_1_segmentation():
    def __init__(self):
        self.input_path = Path("/input/input_coco.json") 
        self.output_file = Path("/output/metrics.json")
    
    def load_json_files(self):
    	
        gt_annotation = open('/opt/app/ground-truth/ground_truth_segmentation.json')
        gt_annotation = json.load(gt_annotation)
        pred_annotation = open(self.input_path)
        pred_annotation = json.load(pred_annotation)
        return gt_annotation, pred_annotation
    
    def matching_annotations(self,gt_annotation, pred_annotation):
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
    
    def calculate_f1_score(self, gt_mask, pred_mask):
        pred_list = []
        for i in range(0,len(pred_mask),2):
            pred_list.append([pred_mask[i],pred_mask[i+1]])
        gt_list = []
        for i in range(0,len(gt_mask),2):
            gt_list.append([gt_mask[i], gt_mask[i+1]])
        checker = False
        try:
            
            intersection = Polygon(pred_list).intersection(Polygon(gt_list))
        except: 
            try:
                pred_polygon =  Polygon(pred_list).buffer(0)
                gt_polygon =  Polygon(gt_list).buffer(0)
                intersection = pred_polygon.intersection(gt_polygon)
                checker = True
            except:
                return 0
        
        if str(type(intersection)) == "<class 'shapely.geometry.polygon.MultiPolygon'>":
            
            intersection = MultiPolygon([x for x in intersection.geoms if str(type(x)) == "<class 'shapely.geometry.polygon.Polygon'>"])

        tp = intersection.area
        
        if tp == 0:
            return 0
        if checker == True:
            fp = (Polygon(pred_list).buffer(0) - intersection).area
            fn = (Polygon(gt_list).buffer(0) - intersection).area
        else:
            fp = (Polygon(pred_list) - intersection).area
            fn = (Polygon(gt_list) - intersection).area
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2*(precision * recall)/ (precision + recall)
        return f1_score

    def mean_f1_score(self,gt_annotation, pred_annotation, list_images_information):
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
                                    class_f1_score = self.calculate_f1_score(k['segmentation'][0], z['segmentation'][0])
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

    def evaluate(self):
        gt_annotation, pred_annotation = self.load_json_files()
        gt_annotation, pred_annotation, list_images_information = self.matching_annotations( gt_annotation, pred_annotation)
        total_mean, list_of_all_means,total_info = self.mean_f1_score( gt_annotation, pred_annotation, list_images_information)
        
        
        whole_info = {}
        whole_info['total mean'] = total_mean
        whole_info['all means per image'] = list_of_all_means
        whole_info['total info'] = total_info
        with open(f'/output/metrics.json', 'w', encoding='ascii') as f:
                    json.dump(whole_info,f)
        print(whole_info)
        return whole_info


if __name__ == "__main__":
    
    Arcade_phase_1_segmentation().evaluate()
