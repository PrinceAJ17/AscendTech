This will be a document/text file covering my thought process throughout the whole assignment.



We first ask ourselves what exactly is the question and what are we trying to achieve here? 



***Develop an AI-powered application that allows users to virtually try on different eyeglasses using a live webcam. The application should provide a realistic experience with accurate glasses positioning, smooth facial tracking, and minimal processing delay.***



Stemming from the main question we try to ask and answer a few sub questions to develop an idea of where we want to go with this:

Q1) What sort of approach can we take with a project like this?

A1) The doc has specified two approaches. Option A would be to build everything from the ground up while still using libraries to not have to reinvent the wheel. This can be a good approach but the results may not be the best considering we would have to build parameters from the ground up. This could be feature engineering, differentiating between image classification/segmentation/object(feature) detection and building up the math involved to determine all of this which are things we would have to simultaneously do while building a software that allows people to try on glasses virtually. 



Key Requirements (as stated in the doc):

● Real-time face detection and facial landmark tracking. -> We would need to build up a object detection model that implements a bounding box correctly (identifying and localizing the object) WHILE having to do feature extraction from an input image (with the help of a neural network). 

● Accurate glasses placement based on facial geometry. -> To implement this we would need to first train a model on faces which could be done using image classification networks like ResNet50 fine tuned on a number of faces and then get some sort of an vector embeddings (eigenfaces) which could describe the shape of the faces to eventually understand where the glasses are supposed to be placed (again with some sort of locatiilisation of the nose bridge, eyes, etc.). This is a very hefty task when you consider pre processing where you would need access to a database of cropped faces, training a CNN on these faces and then implementing the math required for vector embeddings. 

● Realistic rendering of eyeglasses. -> This requires practical aspects of visual rendering of objects (maybe prototyping with extra assets?? I'm not too sure I will need to do further research on this).

● Smooth tracking during head movement. -> There may be limiting factors to this because even with YoLov8 thr tracking was not the smoothest but this may heavily depend on the CPU Cores I will need to reconfirm this.

● Optimized performance with minimal latency. -> This might again depend on system configs but I will need to research this more. This is also a big requirement of the project 



As much as I would love to do this from the ground up, I think a pretrained YOLO model could very well do something of this grandeur while doing it justice by not compromising on performance. I will be going with option B. This is because a big part of the virtual try on relies heavily on accurately detected faces and facial features while the adapting of the glasses itself could easilt be done manually without 



Q2) Now that we have the approach done, let us figure out how we get these glasses to even appear on the faces. Let us take the example of virtual meeting applications like Zoom, Google Meets, etc.

A2) 

