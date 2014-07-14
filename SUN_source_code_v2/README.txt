This software is available for only non-commercial use.  See the attached license in LICENSE.txt under code/scene_sun folder.

This is a post-CVPR version of the code and may produce slightly different (hopefully better) results than the performance reported in the following paper:

J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba
SUN Database: Large-scale Scene Recognition from Abbey to Zoo
Proceedings of 23rd IEEE Conference on Computer Vision and Pattern Recognition (CVPR2010) 

This is the 2nd version of the code, which resolves a few bugs in the 1st version. We would like to thank Mohammadhadi Kiapour for helpful bug report.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

USAGE

There are two top level folders in this zip file: code and data. All source code goes to code folder, and all data goes to code folder. Our source code is located under the folder code/scene_sun. To ease the deployment of the software, we also include the libraries from other research groups in this package under different folders under code folder. Please read their respective compilation and installation instructions as well as software license in their respective folders. Folder data/vocabulary contain k-means clustered codebook for bag-of-word features. We don't provide codes to generate this codebook but it should be straightforward to compute it using our feature extraction code and k-mean in MATLAB toolbox. Although it is possible to put image data and intermediate results in a separate folder, we recommend you to put them inside data/dataset_name .

Step 1: Download and uncompress the image data to the folder data/scene_15class/image or data/scene_397class/image depends on which database you use.
You can download the 15 scene categories dataset from:
http://www-cvr.ai.uiuc.edu/ponce_grp/data/index.html#scenes
and the SUN 397 class dataset from 
http://groups.csail.mit.edu/vision/SUN/
In the folder data/scene_15class and data/scene_397class, we have contained MAT file split10.mat for random training and testng data split that we used for the paper.

Step 2: You need to change the folder path in compute_feature.m and run_kernel_svm.m to select which database for you to use.

Step 3: Run compute_feature.m to compute the image features. You can use more than one MATLAB sessions to run this scripts at the same time parallelly.

Step 4: Run run_kernel_svm.m to train SVM and do evaluation.

Notice: This code is not optimized for speed. It may takes a long time to compute the result. Please use a fast computer with huge memory and be patient. We suggest you to run the code on the 15-scene dataset for debugging purpose to make sure this code works correctly first, before you run on the SUN 397 database. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ENVIRONMENT

We only tested our code in CSAIL Debian Linux on 64bit Intel Machine with 32GB memory, and Matlab Version 7.7.0.471 (R2008b).
But it should be quite straightforward to port the code into Windows or Mac.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

BUG REPORT

Please email to jxiao@csail.mit.edu for reporting any bug. We can only provide minimal support for academic researchers only.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

REFERENCES

This code implements the following papers.  Note that the implementation may not be exact.
Please cite this paper if you use the code in your research.

J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba
SUN Database: Large-scale Scene Recognition from Abbey to Zoo
Proceedings of 23rd IEEE Conference on Computer Vision and Pattern Recognition (CVPR2010) 
