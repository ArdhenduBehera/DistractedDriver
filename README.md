This work is under review in IEEE Transaction on Intelligent Transportation System. 

Title: Deep CNN, Body Pose and Body-Object Interaction Features for Drivers' Activity Monitoring
#### PREREQUISITES ####

Anaconda Python 3.6 or higher

CUDA Toolkit v9.0

CuDNN v7.0.5

TensorFlow 1.8 (GPU)

Keras 2.1.3

AlphaPose https://github.com/MVIG-SJTU/AlphaPose 

OpenPose https://github.com/CMU-Perceptual-Computing-Lab/openpose

Object detection module (faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017) http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz

YOLOv3 https://pjreddie.com/darknet/yolo/

scikit-learn


#### Human Pose Estimation and Object Detector #####

Download and install the human pose and object detectors as mentioned above. Then run 
the coe "object_detector_multi_file"/"object_detector_single_file" to detect object. 
Similarly, run the AlphaPose ("run_driver_DA" or "run_state_farm" for the respective 
"Distracted Drivers" or "State Farm" dataset) for human pose estimation in COCO format.

#### PREPROCESING ####
Preprocess these detected objects ("object_list_train_data") and bodypose ("yamlSingleFileBatchRunTest"). Then 
compute the pairwise realtions.

#### CNN features ####
The CNN features are extracted using script "extract_feature_vgg16" and "extract_features_inception_v3" for 
the respective VGG16 and Inception-V3 features. Similarly, the Inception ResNet-V2 features can be extracted 
using "extract_features_inception_v3" by changing the model.

We have extracted all features and plan to make it available though open-source tools (e.g. github), 
as well as through our institute webpage since the size of files are very large. Please see the link 
https://github.com/ArdhenduBehera/DistractedDriver/

##### Datasets ####

We have used two datasets StateFarm and "Distracted Drivers". Both datasets have the same number of activities. 
For StateFarm dataset, we have two sets: A and B. The set A is the original training set used in Kaggle competition. 
The set B is the test set for the competition. We have annotated the set B. Our evaluation involves training on set A 
and testing on set B and vice versa. 

We followed the train and test procedure provided in "Driver Distraction" dataset. 

##### State-of-the-art deep models #####

The state-of-the-art deep models used for the evaluation (e.g. NASNet, DenseNet, Inception-V3) are included. The 
files are named as the model name.


#### Baseline SVM evaluation ####

The svm evaluations including plat calibration and estimating best C is included in the "baseline_cross_validation"


#### Multi-stream Deep Fusion Network (MDFN) ####
The proposed MDFN is described in "deep_three_stream_DA_fine_tune" and "deep_three_stream_SF_model3" for the 
respective "Distracted Drivers" and "State Farm" dataset. The model is fine-tuned to find the best batch size and 
trained for 100 epochs. Various learning rate and optimizers ("Adam", "RMSProp") are tried for the best performance.

#### MDFN best combinations ####
StateFarm dataset: 
Batch size = 128, Optimizer = Adam, Learning Rate 0.00001, 
Streams: Inception-V3 CNN features, Body pose and body-object interaction
Number of epochs = 100

Distracted Driver Dataset:
batch size = 16, Optimizer = Adam, Learning Rate 0.00001, 
Streams: Inception-V3, VGG16 and Inception ResNet V2 (all CNN features)
Number of epochs = 100

#### Fine-tunning the state-of-the-art models #####
Batch size = 16
Optimizer = RMSProp
Learning Rate = 0.001
Number of epochs = 50

#### Performance Evaluation #####

The script "res_metric" is used for evaluation various metric. It is also used for classifier level fusion.
