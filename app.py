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
st.write("yo yo im ben")

## display 29 images
r1,r2 = st.columns(15),st.columns(15)
rows = r1+r2
for space, image, sign in zip(rows,images, signs):
    with space:
        st.write(sign)
        st.image(image)


upload = st.file_uploader(label='Upload an Image',type=['jpg','png'])

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
#mistakes e[a],g[h],i[v], k[w], m[b],n[b].s[z],v[w], x[z]
