# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

[//]: # (Image References)

[image1]: ./Report/SummaryOfDataTrainingSet.png "SummaryOfDataTrainingSet"
[image2]: ./Report/SummaryOfDataValidationSet.png "SummaryOfDataValidationSet"
[image3]: ./Report/SummaryOfDataTestSet.png "SummaryOfDataTestSet"
[image4]: ./Report/SpeedLimit_20km.png "Traffic Sign 1"
[image5]: ./Report/SpeedLimit_30km.png "Traffic Sign 2"
[image6]: ./Report/SpeedLimit_50km.png "Traffic Sign 3"
[image7]: ./Report/Roadwork.png "Traffic Sign 4"
[image8]: ./Report/AheadOnly.png "Traffic Sign 5"
[image9]: ./Report/grayscale1.png "Gray Scale Sample"
[image10]: ./Report/grayscale2.png "Gray Scale Sample2"
[image11]: ./Report/grayscale3.png "Gray Scale Sample3"
[image12]: ./Report/normalize1.png "Normalize Sample1"
[image13]: ./Report/normalize2.png "Normalize Sample 2"
[image14]: ./Report/normalize3.png "Normalize Sample 3"

[testImage]: ./Report/NewImage.png "List of New Image"



## The following link is to my [Traffic Sign Classifer Project](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used Udacity provided training, validation, and test set of image

#### 0. Load The Data

```python
training_file = 'train.p'
validation_file= 'valid.p'    
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
```

#### 1. Provide a basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic signs data set:

```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid) 

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(pd.Series(y_train).unique())

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

* The size of training set is ?
	Number of training examples = 34799
	
* The size of the validation set is ?
	Number of validation examples = 4410
	
* The size of test set is ?
    Number of testing examples = 12630
	
* The shape of a traffic sign image is ?
    Image data shape = (32, 32, 3)
	
* The number of unique classes/labels in the data set is ?
    Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed for each 43 classifier for German Traffic sign.

![alt text][image1]

![alt text][image2]

![alt text][image3]


Here is an exploratory visualization of the data set. I picked 15 sample for each 43 classifier to display for German Traffic sign.

```python

Sign_Per_Line = 15
trainSignImg = {}

for (signimg,label) in zip(X_train, y_train):
    trainSignImg.setdefault(label, [])
    if len(trainSignImg[label]) < Sign_Per_Line:
        trainSignImg[label].append(signimg)
    
for label in sorted(trainSignImg.keys()):
    plt.figure(figsize=(25,25))
    print(sign_names[label])
    for i in range(Sign_Per_Line):
        plt.subplot(1,Sign_Per_Line,i+1)
        plt.imshow(trainSignImg[label][i])
    plt.show()
```

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


As a first step, I decided to convert the images to grayscale because in image recognition, it is often the method used to convert color images to grayscale has little impact on recognition performance. Since our purpose is recognized the traffic sign, there will be less impact of recognition performance. It is also able to perform under less powerful CPU and/or GPU system. I thought it would be quite useful technique. 

Here is an example of after grayscaling.

![alt text][image9]

![alt text][image10]

![alt text][image11]

```python

# Convert RGB to Grayscale and reshape the size of image to 32x32
def grayscale(imageV):
    grayImageV = []

    for i in range(0, len(imageV)):
        grayImageV.append(cv2.cvtColor(imageV[i], cv2.COLOR_RGB2GRAY))
            
    # plot sample gray image 
    plt.subplot(1, 5, 5)
    plt.imshow(np.reshape(grayImageV[0], (32, 32)), cmap='gray')
    plt.show()
    # plot sample gray image
    
    imageV = np.reshape(grayImageV, (-1, 32, 32, 1))
    return imageV
```	
	
As a last step, I normalized the image data because it amplify the excited neuron while dampening the surrounding neurons. You can either normalize within the same channel or you can normalize across channels. When you are normalizing within the same channel, it’s just like considering a 2D neighborhood of dimension N x N, where N is the size of the normalization window. You normalize this window using the values in this neighborhood. If you are normalizing across channels, you will consider a neighborhood along the third dimension but at a single location. You need to consider an area of shape N x 1 x 1. Here 1 x 1 refers to a single value in a 2D matrix and N refers to the normalization size.

Here is an example of an augmented image:

![alt text][image12]

![alt text][image13]

![alt text][image14]

```python
# Normalization:
def normalize(imageV):
    normalImageV = []
    
    for i in range(0, len(imageV)):
        normalImageV.append(cv2.normalize(imageV[i], imageV[i], 0, 10, norm_type=cv2.NORM_MINMAX))
        
    # plot sample normalized image 
    # plt.subplot(1, 5, 5)
    # plt.imshow(np.reshape(normalImageV[0], (32, 32)), cmap='gray')
    # plt.show()    
    # plot sample normalized image 
    
    return imageV
```	


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the modify version of LeNet (Yann Lecun) from Lesson 9. Convolutional Neural Network. Input will be preprocessed with grayscale and normalization. And the output will be 43 German Traffic Sign Classifier. 

My final model consisted of the following layers:

   | Order of Layer| Layer         		   |     Description	        				   | 
   |:--------------|:---------------------:|:---------------------------------------------:| 
   |               | Input         		   | 32x32x1 RGB image   						   | 
   | Layer 1       | Convolution 3x3       | 1x1 stride, same padding, outputs 28x28x6 	   |
   |               | RELU				   |											   |
   |               | Max pooling	       | 2x2 stride,  outputs 14x14x6  				   |
   | Layer 2       | Convolution 3x3	   | 1x1 stride, same padding, outputs 10x10x16    |
   |               | RELU				   |											   |
   |               | Max pooling	       | 2x2 stride,  outputs 5x5x16    			   |
   |               | Flatten               | Output = 400                                  |
   | Layer 3       | Fully connected	   | Output = 120 								   |
   |               | RELU				   |											   |
   | Layer 4       | Fully connected	   | Output = 84  								   |
   |               | RELU				   |											   |
   | Layer 5       | Fully connected	   | Output = 43  								   |


```python 
### Define your architecture here.
### Feel free to use as many code cells as needed.
def LeNet_5(x):    
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2 )

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits
```

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an following epoch, batch size and learning rate.
I increase epochs to 100 and lower the learning rate to have fine tune the Neural Network
Increase epochs alone could over fitting the network, I also lower the rate.  

```python 
EPOCHS = 50
BATCH_SIZE = 128
rate = 0.001
```


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

```python 
logits = LeNet_5(x)

# Compare logits to the ground-truth labels and calculate the cross entropy
# Cross entopy is a measure how different the logits are from the ground-truth training labels
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)

# Average the cross entropy from all the training images
loss_operation = tf.reduce_mean(cross_entropy)

# Use Adam algorithm (alternative of stochastic gradient descent)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)

# Optimizer uses backpropagation to update the network and minimize training loss
training_operation = optimizer.minimize(loss_operation)
```

My final model results were:
* training set accuracy of ?	
	Training Set Accuracy = 0.996
	
* validation set accuracy of ? 
	Validation Set Accuracy = 0.940
	
* test set accuracy of ?
	Test Set Accuracy = 0.905

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    When I first designed architecture, I didn't pre-processing image with gray scale or normalization. I just chooses to train with LeNet lab for MNIST. 
	Since LeNet lab for MNIST with 10 Classifier can modify and improve to work with Traffic Sign with 43 classifier.
	
* What were some problems with the initial architecture?
	In the beginning, I wasn't able to obtain 93% accuracy which require by lab. After I apply normalization, my accuracy drop even more.
    So I went back and review all the previous lessons. After I included the gray scale I was able to increase the accuracy to 94%. 	
    
* How was the architecture adjusted and why was it adjusted? 
	I adjusted various different pre-processed image method to better fit the previous defined architecture from LeNet lab for MNIST
	
* Which parameters were tuned? How were they adjusted and why?
	I want to fine tune so I lower the learning rate from 0.001 to 0.0001 and increased epoch from 50 to 100 to training more.
	But increasing epoch from 50 to 100 didn't give me the better result. The accuracy of training set didn't get better after epoch 56.
	And stayed at 90%. I believe lowering the learning rate to 0.0001, under-fit the training. 
	
If a well known architecture was chosen:
* What architecture was chosen?
	LeNet_5
	
* Why did you believe it would be relevant to the traffic sign application?
	I learn that the LeNet lab for MNIST with 10 Classifier work pretty well so I believed Traffic Sign with 43 classifier should work pretty as well. 
	It was good learning experience, how to apply what I learn from LeNet lab for MNIST to Traffic Sign Classifier Lab
	
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
	I obtained above 95% accuracy on training and validation. so I was expected to work better then result.
	But I found out choosing test image was crucial as well. At first, I pick any image from google, but I found out it didn't work.
	I had to google better quality of picture with correct format.
 

### Test a Model on New Images

#### 1. Choose Seven German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are Seven German traffic signs that I found on the web:

![alt text][testImage] 	

The Last image might be difficult to classify because the actual image wasn't square. I didn't re-image to created square so, my training neural network might not able to predict properly

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|  Image			                            |     Prediction	        					|  Correctness  |
|:---------------------------------------------:|:---------------------------------------------:| :------------:|
|  Speed limit (30km/h)	                        |  Speed limit (30km/h)							|       O       |
|  Bumpy road  			                        |  Bumpy road           						|       O       |
|  Ahead only					                |  Ahead only									|       O       |
|  Turn left ahead              	      		|  Yield                     					|       X       |
|  Speed limit (50km/h)                         |  Speed limit (50km/h)                         |       O       |
|  Right-of-way at the next intersection        |  Right-of-way at the next intersection        |       O       |
|  Priority road                                |  No entry                                     |       X       |


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71.4%. This compares favorably to the accuracy on the test set of 87.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (30km/h) sign (probability of 1.0), and the image does contain a Speed limit (30km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (30km/h)  						| 
| 0.0     				| Speed limit (50km/h) 							|
| 0.0					| General caution   				     		|
| 0.0	      			| Wild animals crossing			 				|
| 0.0				    | Speed limit (20km/h)       					|

For the second image, the model is relatively sure that this is a Bumpy road (probability of 0.87), but the image contain a Bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.87         			| Bumpy road                					| 
| 0.13     				| Bicycles crossing						        |
| 0.0					| Traffic signals 				        		|
| 0.0	      			| No vehicles   			 	     			|
| 0.0				    | Road work                      				|

For the third image, the model is relatively sure that this is a Ahead only (probability of 1.0), and the image does contain a Ahead only sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Ahead only                					| 
| 0.0     				| Speed limit (30km/h)					        |
| 0.0					| Road work        				        		|
| 0.0	      			| Road narrows on the right	 	     			|
| 0.0				    | Bicycles crossing             				|

For the fourth image, the model is relatively sure that this is a Yield sign (probability of 0.98), and the image does contain a Turn left ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.98         			| Yield                     					| 
| 0.02    				| End of no passing by vehicles over 3.5 metric |
| 0.0					| End of speed limit (80km/h)	        		|
| 0.0	      			| Turn left ahead     		 	     			|
| 0.0				    | Ahead only                    				|

For the fifth image, the model is relatively sure that this is a Speed limit (50km/h) sign (probability of 1.0), and the image does contain a Speed limit (50km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (50km/h)       					| 
| 0.0     				| Speed limit (100km/h)					        |
| 0.0					| Speed limit (30km/h)			        		|
| 0.0					| Speed limit (80km/h)			        		|
| 0.0					| Speed limit (60km/h)			        		|

For the sixth image, the model is relatively sure that this is a Right-of-way at the next intersection sign (probability of 0.99), and the image does contain a Right-of-way at the next intersection sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Right-of-way at the next intersection			| 
| 0.0				    | Pedestrians                   				|
| 0.0   				| Children crossing 					        |
| 0.0					| Road narrows on the right 	        		|
| 0.0	      			| General caution           	     			|


For the seventh image, the model is relatively sure that this is a Yield (probability of 0.82), but the image contain a Priority road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82         			| Yield                      					| 
| 0.13				    | Ahead only                    				|
| 0.02    				| Road work          					        |
| 0.02					| Turn right ahead   			        		|
| 0.0	      			| Priority road                 	 	     			|




