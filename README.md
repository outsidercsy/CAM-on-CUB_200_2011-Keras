# CAM-on-CUB_200_2011-Keras
Implementation of classification activate mapping on CUB-200-2011,using Keras.
Environment:
    python2.7
    Keras2.1.2
    tensorflow1.1.0
    cv2.numpy
    
Usage Instructions:
    First download 'cam_model.h5' at 'https://pan.baidu.com/s/1qWO0K3zSbq3FDnpzCC3xPQ' and put it in the same directory with 'cam_predict.py'.The files, './cam_model.h5' and './fcn_w.npy' are the already trained model. So with environment installed, you can directly run 'cam_predict.py', the result will be saved in the file './bounding_box_generate'.
    If you want to train the model yourself, first download CUB-200 dataset from the offcial website. And create file folders './CUB_200_2011/CUB_200_2011/train', './CUB_200_2011/CUB_200_2011/validation' and './CUB_200_2011/CUB_200_2011/segmentations'. And then fill these folders with the dataset you have just downloaded. Note that because we have used ImageDataGenerator of Keras, the train dataset must be arranged in a certain way. With train dataset prepared, run 'cam_train.py'. And the file 'cam_traincontinue.py' is for additional training epoches if needed. 
    
If having any question when using this code, email me at U201513500@hust.edu.cn.
    
 
    

    
