# face-detection
Product:
Our System will capture the image of the person and recognize his face with the pictures present in our trained data, if the image is recognized then then person is granted permission otherwise denied. Our system has the capability of self-training on the basis of the daily changes occurring in face of person registered in our system. Our system uses Feature based approch for face detection.

Background:
Face detection can be characterized into two parts such as, feature based and holistic approaches techniques. The most commonly used technique in face detection was feature based which was used to characterize a low-dimensional face representation in order to identify proportions of separations, zones, and various angles. The face representation used by this technique is pretty much attractive but this technique does not provide accurate results. 

Working:
First we have to get multiple images of multiple person for the training of our system. From that images our system detects the faces and start processing. Once the face is detected, it is preprocessed the faces present in the images and create normalized and fixed-point input for neural network. Now neural network plays its role of feature extractor here and creates low-dimensional representation of the face which identifies a person’s face. The low-dimensional representation is basically the used in classifiers for classification of one’s face or we may say that it can also be used for clustering. and finally it will generate classifier.pkl file.

To run our system need some installation:
Conda Installation
Python Installation
	Intall Required packages
	Download Python 2.7.13
	Extract
	Compile Python Source
Check Python Version
Dlib:
	Install dlib prerequisites
o	Boost
o	Boost.Python
o	X11/XQuartx
Alternate
OpenFace
	Installing Dependencies
	Installing Openface
	Now test Openface	

Commands for training:
- Type Sudo su
- then enter system password
- Type command	./util/align-dlib.py ./train_70/ align outerEyesAndNose ./aligned_70/ --size 96
	Press enter (it will start transformation process)
- Type Command ./batch-represent/main.lua -outDir ./generate_70/ -data ./aligned_70/
   Press enter (it will start Embedding process)
 -Type command 	./demos/classifier.py train ./generate_70/ 
   Press enter (it will start Training process)
   System is now successfully Trained.
  
 Command for Face recognition:
Type Command ./demos/classifier.py infer ./generate-embeddings/classifier.pkl aligned_face.jpg

