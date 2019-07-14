# DLCVizsla
My NTU CommE 5052 Deep Learning for Computer Vision (by Prof. Frank Wang) homeworks,  
you can get more details in each ```README.md``` or ```report.pdf``` inside folders.   
> The Vizsla (Hungarian: [ˈviʒlɒ]) is a dog breed originating in Hungary.  
<img src="https://www.pets4homes.co.uk/images/breeds/88/large/34aaa9d6aa84f3926b461f88e4dcce51.jpg" width="350">  

---

* project1_hdrImage
  * Alignment (MTB + Image Pyramid + Offset Search)
  * HDR (recover CRF + generate radiance map)
  * Tone mapping (opencv or Photomatix)
* project2_panorama
  * feature detection (SIFT detector, SIFT discriptor)
  * feature matching (Brute force(2-norm distance), flann(kd-tree, knn-search))
  * image matching (RANSAC finding shift)
  * stitching n blending (Linear filter on fixed width edge or entire overlapRegion, Naive overlap stitching)
  * end to end alignment (Scattering y displacement)
  * cropping
* project3_matchMove
  * structure from motion (sfm) concept
  * Blender, Voodoo, iMovie, Photoshop
* VFX_Final
  * Visual odometry with deep learning (DeepVO)
  * https://github.com/shannon112/VFX_Final

## contents:  
* 01.PCA_kNN - face recognition
  * cv2&matplotlib(gray), MSE, PCA, reconstruction, k-NN  
* 02.BoW - image classification
  * cv2&matplotlib(rgb), Patches, k-means, PCA, scatterPlot3D, Soft-max(max pooling), k-NN
* 03.MNISTdemo - handwritten digit recognition
  * for pytorch and cuda testing
