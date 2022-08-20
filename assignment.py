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

#2.4 Tensorflow Model Implementation
#Tensorflow models are built layer by layer. A layer's input dimensions (sin above) are calculated for you. You specify a layer's output dimensions and this determines the next layer's input dimension. The input dimension of the first layer is derived from the size of the input data specified in the model.fit statment below.

#Note: It is also possible to add an input layer that specifies the input dimension of the first layer. For example:
#tf.keras.Input(shape=(400,)),    #specify input shape
#We will include that here to illuminate some model sizing.

#Below, using Keras Sequential model and Dense Layer with a sigmoid activation to construct the network described above.


model = Sequential(
    [               
        tf.keras.Input(shape=(400,)),    #specify input size
        ### START CODE HERE ### 
        Dense(25,activation='sigmoid'),
        Dense(15,activation='sigmoid'),
        Dense(1,activation='sigmoid')
        
        
        ### END CODE HERE ### 
    ], name = "my_model" 
)     

model.summary()

#Model: "my_model"
#_________________________________________________________________
# Layer (type)                Output Shape              Param #   
#=================================================================
# dense (Dense)               (None, 25)                10025     
#                                                                
# dense_1 (Dense)             (None, 15)                390       
                                                                 
 #dense_2 (Dense)             (None, 1)                 16        
                                                                 
#=================================================================
#Total params: 10,431
#Trainable params: 10,431
#Non-trainable params: 0

#To run the model on an example to make a prediction, use Keras predict. The input to predict is an array so the single example is reshaped to be two dimensional.

#2.5 NumPy Model Implementation (Forward Prop in NumPy)

#Exercise 2
#Below, build a dense layer subroutine. The example in lecture utilized a for loop to visit each unit (j) in the layer and perform the dot product of the weights for that unit (W[:,j]) and sum the bias for the unit (b[j]) to form z. An activation function g(z) is then applied to that result. This section will not utilize some of the matrix operations described in the optional lectures. These will be explored in a later section.

def my_dense(a_in, W, b, g):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      a_out (ndarray (j,))  : j units
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):             
        w = W[:,j]                     
        z = np.dot(w, a_in) + b[j]     
        a_out[j] = g(z)                
    return(a_out)

#2.6 Vectorized NumPy Model Implementation (Optional)
#Below, compose a new my_dense_v subroutine that performs the layer calculations for a matrix of examples. This will utilize np.matmul().

# UNQ_C3
# GRADED FUNCTION: my_dense_v
​
def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer
    Args:
      A_in (ndarray (m,n)) : Data, m examples, n features each
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (1,j)) : bias vector, j units  
      g    activation function (e.g. sigmoid, relu..)
    Returns
      A_out (ndarray (m,j)) : m examples, j units
    """
### START CODE HERE ### 
    z = np.matmul(A_in,W) + b
    A_out = g(z)
        
    
### END CODE HERE ### 
    return(A_out)

    
















