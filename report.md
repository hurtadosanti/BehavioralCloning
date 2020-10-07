# **Behavioral Cloning** 

**Behavioral Cloning Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Solution Design Approach

The process to train the model was:

- First load the data from the log, we have 8036 rows with 'center', 'left', 'right' image paths ,and  the floating value for the 'steering'
- Shuffle and split the log file
- Load and preprocess each batch of data using a generator
- Load and preprocess each training batch using a generator for the validation set. The validation set does not flip the images.

### Training Set & Training Process
To train the network we use three laps with the data provided with the project

## Model Architecture and Training
http://alexlenail.me/NN-SVG/LeNet.html
### Model architecture development strategy
We develop three models for the network:
- Simple network to test out the different configurations of the keras layers
- LeNet5 configuration to train and validate and learn how different configurations affect the results
- Nvidia model adding dropout layers to lower the overfitting, finally we use the test data to validate the loss of the model
### Reducing overfitting in the model
To reduce the overfitting of the model we use the following strategies:
- Since the data is steering is bias towards to the right we flip the images from right to left and get the inverse steering measure by multiply by -1
- We randomly set the gamma of the images 
- Use the left and right cameras and compute a +/- 0.25 correction for the steering to the right and left cameras, so for each frame we have three images and three different steering measures. This helps the network to know how to recover when driving to the sides
- Use a dropout layers of 0.25 to the model, improving the loss measure and reducing overfitting.

### Model parameter tuning
- We use batch_size = 32 x 2(side cameras)+1(flip image)
- We use the Adam optimizer with a learning_rate = 0.0003 that gives a proper loss improvement per epoch
- For the loss we use Mean Absolute Error since we are solving a regresion problem
- To learn what parameters fit best we use 15 epocs, however most of the time 10 is the best number before the loss begins to increase again
- We use a dropout 0.25 to improve the loss measure

### Training data
The data provided was divided in 60% training, 20% validation and 20% test data. 

To improve performance and readibility we use the text data only for shuffling and splitting, afterwords the generators only have to read the images from the paths specified.

The data was load into memory using generators to load only the batch images into memory. 


### 2. Final Model Architecture
|Layer|Parameters |Activation|Output Shape|Weight parameters   | 
|-----|----------|----------|------------|---|
|Lambda|Normalization||(160, 320, 3)|0|
|Cropping|70,25 pixels||(65, 320, 3)|0|    
|Convolution (filters, kernel_size, stride)|24,5,2|Relu|(31, 158, 24)|1824|
|Convolution (filters, kernel_size, stride)|36,5,2|Relu|(14, 77, 36)|21636|
|Convolution (filters, kernel_size, stride)|48,5,2|Relu|(5, 37, 48)|43248|
|Dropout|0.25| |(5, 37, 48)|0|
|Convolution (filters, kernel_size, stride)|64,3,1|Relu|(3, 35, 64)|27712|     
|Convolution (filters, kernel_size, stride)|64,3,1|(1, 33, 64)||36928|
|Flatten| | |(2112)|0|
|Dense|100|(100)||211300|
|Dropout|0.25||(100)|0|         
|Dense|50|(50)||5050|
|Dense|10|(10)||510|
|Dense|1|(1)||11|        


Total params: 348,219

Trainable params: 348,219


