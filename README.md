# BIRD CLASSIFICATION CHALLENGE

## TASK

Our objective is to develop a machine learning model capable of classifying 555 bird species and compete in [the Kaggle Competition](https://www.kaggle.com/competitions/birds23wi "Birds Birds Birds Are they real?")

## Data Set

We have been provided with a dataset comprising 38,561 images of 555 distinct bird species to train our machine learning algorithm. Additionally, a separate subset of 10,000 images has been designated for testing without labels.

#### Some Sample Images



## Preprocessing and Augmentation

We conducted a series of tests to evaluate the impact of different combinations of image transformations on the performance of the RESNET 18 model. These transformations included random cropping, flipping, rotation, Gaussian noise, color inversion, and color jitter. We aslo performed batch normalization. Our testing approach was identical to that used in [Tutorial 4 - Transfer Learning to Birds.ipynb.](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing), except that we used 20% of the training data as validation to assess the effects of the transformations. No significant increase in accuracy was observed with any of the transformations in the first ten epochs or so. The underlying reason for this could be the subtle nature of the features that differentiate bird species, and the effects of random transformations may not be apparent until the model has been trained for a significantly higher number of epochs.

#### Batch normalizations

Initially, the RESNET 18 model's validation accuracy was approximately 58% when trained without batch normalization. Afterwards, we implemented batch normalization using the mean and standard deviation values recommended in the model documentation, resulting in a significant increase in accuracy to 67%.


## TRAINING and SCORES

Initially, we trained the RESTNET 50 model using 80% of the data, resized to 256 by 256, as recommended in the documentation. Random crop, horizontal flip, color jitter, and adjust sharpness transformations were applied, in addition to batch normalization. For the first ten epochs, we used a learning rate of 0.01, which we reduced to 0.001 for five additional epochs. This approach yielded a score of 0.78 in the Kaggle competition. Subsequently, using the same approach, the EfficientNet v2 small model was trained, resulting in a score of 0.76.

Following this, all transformations were removed, leaving only batch normalization, and the number of images was doubled by mirroring each one. We trained the EfficientNet v2 small model using 256 by 256 images for six epochs, with a learning rate of 0.01, followed by two more epochs with 0.001 learning rate using 384 by 384 images. 80% of the doubled data was used for training, with the remaining data reserved for validation. This approach yielded a score of 0.809 in the competition.

We applied the cross entropy loss function and stochastic gradient descent (SGD) optimizer in each of our training sessions. We also set a momentum value of 0.9 and a weight decay value of 0.0005 across all experiments.

## Challenges

We used Kaggle notebook for training our models, but the limited computational resources and complex interface made the process difficult. We also attempted to train the model on our laptops, but experienced performance issues.

Applying resizing transformations using Kaggle resulted in CPU bottlenecks while training the models. Therefore, we opted to download the dataset and perform the resizing on our own machines before uploading them back to Kaggle. Additionally, using the aforementioned random transformations resulted in further CPU bottlenecks. In our final submission, to expand the dataset without applying transformations, we mirrored each image, ultimately doubling the size of the dataset.


## Discussion and Conclusion




[EfficientNet Traning Notebook](https://www.kaggle.com/code/zeynelgurbuz/zeynel-efficientnet-v2)

[ResNet50 Training Notebook ](https://www.kaggle.com/code/sayujshahi/seventy8accuracy)

