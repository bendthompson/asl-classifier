
# American Sign Language (ASL) Classifier 🧏

A CNN Deep Learning model that can classify 28 different ASL signs with 89% test accuracy

## Streamlit App
Classify ASL alphabet images, and your own uploaded signs using the model app. 

Try it! -[App website](https://asl-classifier-app.herokuapp.com/)

## Skills I used/learned

👌👋 Worked effectively in the cloud to train a large model on 87,000 images 

✌️👏 Added my own model architecture to pretrained layers from the VGG16 model

🤟✋ Used data augmentation in Keras to improve the models ability to generalise

🤙🤘 Used a range of Streamlit features to create a Heroku-hosted frontend where you can use the model on your own images


## Model

I used a Convolutional Neural Network and acheived 89% accuracy on my test set. 
You can see that some signs look very similar to each other, which is where misclassification would occasionally happen.

I trained the model on this dataset from Kaggle. 
It has 87,000 images in 29 classes, with pictures taken in a range of lightings and positions, helping the model be effective at recognising signs in many contexts.

I partly used transfer learning and training my own layers. 
I used VGG16's CNN and added 1 pooling layer, 1 large hidden layer, and 1 output layer.

## Aim

My aim with this project was just to build an effective image classifier using transfer learning on a challenging dataset that I'm interested in.

I chose the ASL alphabet firstly because there were more readily available datasets. 
Secondly, because it would potentially be more of a challenge to classify than some alternatives.

I personally speak British Sign Language (BSL), as my parents are deaf. 
The BSL alphabet is a combination of one-handed signs and two-handed signs. 
While the ASL alphabet consists of only one-handed signs. 
My thinking was that with many similar looking signs (A&E, M&N, G&Z, etc.) it would be more of a challenge to classify.

## Sign Language Processing (SLP)

SLP is a developing field of deep learning I have a personal interest in. 
It's complex and still evolving, but mainly combines both Natural Language Processing and Computer Vision. 

Classifying individual signs is very small step in the long process of SLP! 
Sign language is much more than just a sequence of signs, 
so in the SLP field lots of work and research is going on to capture the meaning expressed through body language, facial expression, sign momentum etc.
