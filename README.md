# DLCVizsla
My NTU CommE 5052 Deep Learning for Computer Vision (by Prof. Frank Wang) homeworks,  
you can get more details in each ```README.md``` or ```report.pdf``` inside folders.   
> The Vizsla (Hungarian: [ˈviʒlɒ]) is a dog breed originating in Hungary.  
<img src="https://www.pets4homes.co.uk/images/breeds/88/large/34aaa9d6aa84f3926b461f88e4dcce51.jpg" width="350">  

---

* demo_digit_recognition
  * MNIST handwritten digit recognition
  * for pytorch and cuda testing
* hw1_face_recognition
  * face images of 40 different subjects and 10 grayscale images for each subject, all of size (56, 46) pixels
  * cv2&matplotlib(gray), MSE, PCA, reconstruction, k-NN  
* hw1_image_classification
  * 4 categories (classes) and 500 RGB images for each category, all of size (64, 64, 3) pixels
  * cv2&matplotlib(rgb), Bag of Word(BoW), Patches, k-means, PCA, scatterPlot3D, Soft-max(max pooling), k-NN
* hw2_YOLOv1_object_detection
  * YOLOv1: Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
  * trained on DOTA-v1.5 Aerial Images 
  * yolo loss, vgg16_bn+linear model
* hw3_dcgan_acgan_dann
  * GAN[1], DCGAN[2], ACGAN[3], DANN[4], GTA[5], tSNE plot
  * [1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
  * [2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
  * [3] Odena, Augustus, Christopher Olah, and Jonathon Shlens. "Conditional image synthesis with auxiliary classifier gans." Proceedings of the 34th International Conference on Machine Learning-Volume 70. JMLR. org, 2017.
  * [4] Ganin, Yaroslav, et al. "Domain-adversarial training of neural networks." The Journal of Machine Learning Research 17.1 (2016): 2096-2030.
  * [5] Sankaranarayanan, Swami, et al. "Generate to adapt: Aligning domains using generative adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.

  * trained on USPS(28,28), MNIST-M(28,28,3), SVHN(28,28,3) Dataset

* hw4_rnn_action_recognition
* DLCV_Final
  * Visual odometry with deep learning (DeepVO)
  * https://github.com/shannon112/VFX_Final

## contents:  
* 01.PCA_kNN - face recognition
* 02.BoW - image classification
* 03.MNISTdemo - 
  
