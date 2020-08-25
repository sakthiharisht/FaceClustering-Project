# FaceClustering-Project
A face clustering project using DBSCAN and Chinese whispers algorithm

1. Create an dataset folder with a set of input images
2. Run encode_faces.py first to encode the images into a pickle file. 
Syntax to run the code using HOG method is as follows:
python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog
3. Then run the cluster_faces.py to cluster using DBSCAN algorithm
Syntax given below:
python cluster_faces.py --encodings encodings.pickle
4. Next run cluster_faces_cw.py to cluster faces using Chinese whispers algorithm
Syntax given below:
python cluster_faces_cw.py --faces_folder_path dataset --output_folder output 

Please clear data from the output folder before running step 4.
