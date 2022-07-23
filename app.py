import streamlit as st

from os.path import join, isfile, dirname
from os import listdir
from keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

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
st.write("yo yo im ben")

## display 29 images
r1,r2 = st.columns(15),st.columns(15)
rows = r1+r2
for space, image, sign in zip(rows,images, signs):
    with space:
        st.write(sign)
        st.image(image)


upload = st.file_uploader(label='Upload an Image',type=['jpg','png'])



if upload:
    #send image to model
    st.image(upload)
    #output prediction
    pass
