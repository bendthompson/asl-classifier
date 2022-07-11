import streamlit as st

from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#tagline
#explain &links
#load example images
#display example images
#input image
#model.predict
#is it correct
#dict image

best_model = load_model('bestmodel.h5')

st.title("ASL Classifier👋")
st.subheader("This is a deep learning app that classifies which American Sign Language alphabet sign is in an image")
st.write("yo yo im ben")


upload = st.file_uploader(label='Upload an Image',type=['jpg','png'])

if upload:
    #send image to model
    #output prediction
    pass
