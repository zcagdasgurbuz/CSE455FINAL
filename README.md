# BIRD CLASSIFICATION CHALLENGE by ZGURBUZ & SSHAHI

## TASK

Our objective is to develop a machine learning model capable of classifying 555 bird species and compete in [the Kaggle Competition](https://www.kaggle.com/competitions/birds23wi "Birds Birds Birds Are they real?")

## Data Set

We have been provided with a dataset comprising 38,561 images of 555 distinct bird species to train our machine learning algorithm. Additionally, a separate subset of 10,000 images has been designated for testing without labels.

#### Some Sample Images

![0a5e03dfa2514b9389335fa56ba6097d](https://user-images.githubusercontent.com/45305014/224865729-f1428d67-df56-4164-9587-afdc6529c6bf.jpg)
![0a6edc1ee981433bae036278ef5aa629](https://user-images.githubusercontent.com/45305014/224865734-a3b0117f-5cc7-4db1-bb53-944e97307460.jpg)
![0a7d2b8616b94ba0a28ddae6de72a285](https://user-images.githubusercontent.com/45305014/224865736-30a652ad-0083-4e61-8758-06a1d74b1245.jpg)
![0a48a3340f02400fb23453245759213a](https://user-images.githubusercontent.com/45305014/224865737-ca96bbbd-f7d7-4e85-945d-d632047f4af5.jpg)
![0a57a1c2ff8c41619288bcf511104ec2](https://user-images.githubusercontent.com/45305014/224865738-d2cc6b93-795e-4749-b28e-4e3ba28b17cc.jpg)
![0a94d110865d46ea8111e4c58e1538fd](https://user-images.githubusercontent.com/45305014/224865741-81e46049-0f4d-4b25-86b2-42fe15922b47.jpg)
![0a254f34050b43a8903a40ff73871ccf](https://user-images.githubusercontent.com/45305014/224865744-cc84629c-50ff-4ef1-84d2-406c05e564d7.jpg)
![0a704e3c28c5472d9fdf814805a82a4a](https://user-images.githubusercontent.com/45305014/224865745-a705411e-f3ab-4c31-825c-44fb7b5e7bc0.jpg)
![0a6083dabcfd486d8d023e06bdd50898](https://user-images.githubusercontent.com/45305014/224865747-cee967ae-ccf0-40cb-9700-5f24ea4efc2d.jpg)

## Preprocessing and Augmentation

We conducted a series of tests to evaluate the impact of different combinations of image transformations on the performance of the RESNET 18 model. These transformations included random cropping, flipping, rotation, Gaussian noise, color inversion, and color jitter. We aslo performed batch normalization. Our testing approach was identical to that used in [Tutorial 4 - Transfer Learning to Birds.ipynb.](https://colab.research.google.com/drive/1kHo8VT-onDxbtS3FM77VImG35h_K_Lav?usp=sharing), except that we used 20% of the training data as validation to assess the effects of the transformations. No significant increase in accuracy was observed with any of the transformations in the first ten epochs or so. The underlying reason for this could be the subtle nature of the features that differentiate bird species, and the effects of random transformations may not be apparent until the model has been trained for a significantly higher number of epochs.

#### Batch normalization

Initially, the RESNET 18 model's validation accuracy was approximately 58% when trained without batch normalization. Afterwards, we implemented batch normalization using the mean and standard deviation values recommended in the model documentation, resulting in a significant increase in accuracy to 67%.


## Training and scores

Initially, we trained the RESTNET 50 model using 80% of the data, resized to 256 by 256, as recommended in the documentation. Random crop, horizontal flip, color jitter, and adjust sharpness transformations were applied, in addition to batch normalization. For the first ten epochs, we used a learning rate of 0.01, which we reduced to 0.001 for five additional epochs. This approach yielded a score of 0.78 in the Kaggle competition. Subsequently, using the same approach, the EfficientNet v2 small model was trained, resulting in a score of **0.76**.

Following this, all transformations were removed, leaving only batch normalization, and the number of images was doubled by mirroring each one. We trained the EfficientNet v2 small model using 256 by 256 images for six epochs, with a learning rate of 0.01, followed by two more epochs with 0.001 learning rate using 384 by 384 images. 80% of the doubled data was used for training, with the remaining data reserved for validation. This approach yielded a score of **0.809** in the competition.

We applied the cross entropy loss function and stochastic gradient descent (SGD) optimizer in each of our training sessions. We also set a momentum value of 0.9 and a weight decay value of 0.0005 across all experiments.

## Challenges

We used Kaggle notebook for training our models, but the limited computational resources and complex interface made the process difficult. We also attempted to train the model on our laptops, but experienced performance issues.

Applying resizing transformations using Kaggle resulted in CPU bottlenecks while training the models. Therefore, we opted to download the dataset and perform the resizing on our own machines before uploading them back to Kaggle. Additionally, using the aforementioned random transformations resulted in further CPU bottlenecks. In our final submission, to expand the dataset without applying transformations, we mirrored each image, ultimately doubling the size of the dataset.


## Discussion and Conclusion

During our project, we observed that training the EfficientNet v2 small model resulted in a higher score than the RESNET 50 model. While we used larger and more images (generated by mirroring the original dataset) while training the EfficientNet, we are unsure what specifically caused the difference in performance. However, this observation does suggest that it might be worth exploring other architectures and techniques to improve the accuracy of our model.

Additionally, while we experimented with various image transformations, including random cropping, flipping, rotation, Gaussian noise, color inversion, and color jitter, we only trained for 10 epochs due to limited computational resources. It is possible that a longer training period, with more epochs, might have produced better results with the use of transformations. This is a topic that could be further explored in future work.

We also observed that random color change transformations might cause misclassification, as they could alter the color of some birds. This is an important consideration when choosing image transformations for bird classification tasks.

Finally, as a future direction, we might consider subscribing to Google Colab+ for a month to experiment with a wider range of models and transformations. This could help us to identify further improvements and insights for bird classification.

## VIDEO

Our presentation is available for viewing on YouTube.

[![Watch the video](https://img.youtube.com/vi/jawDqYXGBr8/maxresdefault.jpg)](https://youtu.be/jawDqYXGBr8)

---

[EfficientNet Traning Notebook](https://www.kaggle.com/code/zeynelgurbuz/zeynel-efficientnet-v2)

[ResNet50 Training Notebook ](https://www.kaggle.com/code/sayujshahi/seventy8accuracy)

----

This website was created by Zeynel Cagdas Gurbuz and Sayuj Raj Shahi as part of their final project submission for the
[CSE 455](https://courses.cs.washington.edu/courses/cse455/23wi/ "The Ancient Secrets of Computer Vision")  course during the winter quarter of 2023.

To reach Zeynel : [zgurbuz@uw.edu](mailto:zgurbuz@uw.edu)  
To reach Sayuj : [sayujs@uw.edu](mailto:sayujs@uw.edu)
