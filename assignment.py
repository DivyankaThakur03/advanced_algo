#Tensorflow and Keras
#Tensorflow is a machine learning package developed by Google. In 2019, Google integrated Keras into Tensorflow and released Tensorflow 2.0. 
#Keras is a framework developed independently by François Chollet that creates a simple, layer-centric interface to Tensorflow. 
#This course will be using the Keras interface.

#2.3 Model representation
#The neural network you will use in this assignment is shown in the figure below.

#This has three dense layers with sigmoid activations.
#Recall that our inputs are pixel values of digit images.
#Since the images are of size 20×20, this gives us 400 inputs


#The parameters have dimensions that are sized for a neural network with 25 units in layer 1, 15 units in layer 2 and 1 output unit in layer 3.

#Recall that the dimensions of these parameters are determined as follows:

#If network has sin units in a layer and sout units in the next layer, then
#W will be of dimension sin×sout.
#b will a vector with sout elements
#Therefore, the shapes of W, and b, are

#layer1: The shape of W1 is (400, 25) and the shape of b1 is (25,)
#layer2: The shape of W2 is (25, 15) and the shape of b2 is: (15,)
#layer3: The shape of W3 is (15, 1) and the shape of b3 is: (1,)
#Note: The bias vector b could be represented as a 1-D (n,) or 2-D (n,1) array. Tensorflow utilizes a 1-D representation and this lab will maintain that convention.
