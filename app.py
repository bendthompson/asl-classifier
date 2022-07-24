import streamlit as st

from os.path import join, isfile, dirname
from os import listdir
import numpy as np
from keras.models import load_model
from PIL import Image
from skimage import transform
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

#tagline
#explain &links
#load example images
#display example images
#input image
#model.predict
#is it correct
#dict image

## page config
st.set_page_config(page_title="asl", page_icon="👋",layout="wide")

## load model
best_model = load_model('bestmodel.h5')

## load 29 images
filedir = 'data/app_examples'
signs = []
images = []
for file in sorted(listdir(filedir)):
    if isfile(join(filedir, file)):
        images.append(Image.open(join(filedir, file)))
        if len(file) > 10:
            signs.append(file[1:-8])
        else:
            signs.append(file[:-8])
imgarrays = [img_to_array(img) for img in images]

## text info
st.title("ASL Classifier👋")
st.subheader("This is an app that classifies which American Sign Language alphabet sign is in an image using deep learning ")
st.write("dataset w variety of images in different lightings and positions")
st.write("I trained the model on a dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet). It has 87,000 images in 29 classes, with pictures taken in a range of lightings and positions, helping the model be robust at recognising signs in many contexts. ")

## display 29 images
heads = st.columns([10,4])
with heads[0]:
    st.subheader('Example signs')
with heads[1]:
    st.subheader('Try it yourself')

c= st.columns([1,1,1,1,1,1,1,1,1,1,4])
cols = c[:10]+c[:10]+c[:10]
with c[10]:
    #st.subheader('Try it yourself')
    upload = st.file_uploader(label='Drag and drop an example or upload your own',type=['jpg','png'])

for space, image, sign in zip(cols,images, signs):
    with space:
        st.write(sign)
        st.image(image)

if upload:
    with c[10]:
        st.subheader('Prediction')
        st.image(Image.open(upload), width=100)



def max2(preds):
    arr = preds.copy()
    idx1 = arr.argmax()
    arr[idx1] = 0

    idx2 = arr.argmax()

    return idx1, idx2


if upload:
    ## send image to model
    input = Image.open(upload)
    input = img_to_array(input)
    input = transform.resize(input, (220, 220, 3))
    input = preprocess_input(input)
    input = np.expand_dims(input, axis=0)
    pred = best_model.predict(input)
    ## output prediction
    idx1, idx2 = max2(pred[0])
    st.write(pred)
    st.write(idx1, idx2)
    st.write(f'Is it {signs[idx1]}? If not is it {signs[idx2]}?')


    pass
