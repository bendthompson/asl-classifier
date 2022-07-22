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
st.set_page_config(page_title="asl", page_icon="👋")
## load model
best_model = load_model('bestmodel.h5')
## load 29 images

filedir = 'data/app_examples'
signs = []
images = []
for file in sorted(listdir(filedir)):
    if isfile(join(filedir, file)):
        signs.append(file[:-4])
        images.append(Image.open(join(filedir, file)))

r1 = st.columns(3)
r2 = st.columns(3)
rows = [r1[0], r1[1], r1[2], r2[0], r2[1], r2[2]]
for space, image, sign in zip(rows,images, signs):
    with space:
        st.image(image)
        st.write(sign)
st.title("ASL Classifier👋")
st.subheader("This is an app that classifies which American Sign Language alphabet sign is in an image using deep learning ")
st.write("yo yo im ben")

## display 29 images


upload = st.file_uploader(label='Upload an Image',type=['jpg','png'])



if upload:
    #send image to model
    #output prediction
    pass
