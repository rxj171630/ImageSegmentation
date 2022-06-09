Link to dataset: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit 


Our code is organized by each segmentation method. There is a folder for each method that contains the code and images that we used for each method.


For threshold segmentation, all of the code is contained in the jupyter notebook. All input images are in the “img” folder and output images are in the “output” folder within the “Threshold Segmentation” folder.


For KMeans Segmentation, all of the code is contained in the jupyter notebook, and all input and output images are in the corresponding folders within the “KMeans_Segmentation” folder.


For Edge Detection Segmentation, download the entire folder titled “edge_detector”, and run the Jupyter notebook that is included within the folder. The test images can be found in the “test_imgs” folder, and the output can be seen when running the notebook.


For Mask RCNN, as noted in the final report, our implementation of Mask RCNN is broken and thus unfinished. All code for our implementation of MaskRCNN is in the “MaskRCNN/src” folder. An attempt to train and run it is in the “MaskRCNN/mask_rcnn.ipynb” notebook. The images in “MaskRCNN/result_images” are results obtained using an online off-the-shelf implementation found on https://colab.research.google.com/github/tensorflow/tpu/blob/master/models/official/mask_rcnn/mask_rcnn_demo.ipynb .


Documentation for the project can be found [here](https://github.com/rxj171630/ImageSegmentation/blob/main/ImageSegmentation.pdf)
