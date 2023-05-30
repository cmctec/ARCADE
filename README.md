# Official Page for ARCADE challenge for MICCAI Conference

![aRCADE__1_-2](https://github.com/cmctec/ARCADE/assets/70023022/c506f8c6-875b-4aa6-a0ec-4b259b3ad794)

## Introduction
Coronary artery disease (CAD) is a condition that affects blood supply of heart, due to buildup of atherosclerotic plaque in the coronary arteries. CAD is one of the leading death causes around the world. The most common diagnosis procedure for CAD is coronary angiography, which uses contrast material and X-rays for observation of lesions in arteries, this type of procedure showing blood flow in coronary arteries in real time what allows precise detection of stenosis and control of intraventricular interventions and stent insertions. Coronary angiography is useful diagnostic method for planning necessary revascularization procedures based on calculated occlusion and affected segment of coronary arteries. The development of automated analytical tool for lesion detection and localization is a promising strategy for increasing effectiveness of detection and treatment strategies for CAD.

There are very few works aimed to segment coronary arteries, and currently it is very costly and time-consuming to manually select segments and stenotic lesions within coronary angiography. This challenge’s purpose is to benchmark coronary artery segmentation as well as stenosis detection methods which could be used to reduce time spent while maintaining high accuracy of coronary angiography analysis. For that, dataset of coronary angiography frames with labelled coronary artery segments and the locations of stenotic plaques are provided.

Coronary artery segments were labelled following Syntax Score definitions. It is expected for models to be able to effectively augment given number of images to increase the robustness.

To sum up the challenge:

1) 2 supervised tasks: coronary artery segmentation and stenosis detection (segmentation) (both tasks are instance

segmentation tasks).

2) 1500 images with labeled coronary arteries and 1500 images with labeled stenosis for training and testing.

3) Evaluation metrics: challenge submissions are evaluated using mean F1 score and will be tested on 300 images for both tasks. Inference time limit is set to be 5 seconds/per image.


## Dataset
Dataset is available for ARCADE challenge participants and available on: https://zenodo.org/record/7981245

## Navigation in pages
Evaluation folder consists of evaluation evaluation script as well as required files and instructions to run evaluation script on your own. [Will be posted on 02.06.2023]
Useful scripts folder consists of scripts that might be helpful with working on annotation formats and etc.
