import streamlit as st

from os.path import join, isfile, dirname
from os import listdir
import numpy as np
from keras.models import load_model
from PIL import Image
from skimage import transform
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input


## page config
st.set_page_config(page_title="ASL Classifier", page_icon="👋",layout="wide")

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


## headers
heads = st.columns([4,10])
with heads[0]:
    st.title("ASL Classifier👋")
with heads[1]:
    st.write('')
    st.subheader("This app uses deep learning to classify American Sign Language images")
    st.markdown('**Built by Ben Thompson, learn more or connect with me on [github](https://github.com/bendthompson) and [linkedin](https://www.linkedin.com/in/ben-d-thompson/)**')

## model&app information
with st.expander('Information'):

    st.markdown("**Model**")
    st.write("I used a Convolutional Neural Network and acheived 89% accuracy on my test set. You can see that some signs look very similar to each other, which is where misclassification would occasionally happen.")
    st.write("""I trained the model on this dataset from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).
             It has 87,000 images in 29 classes, with pictures taken in a range of lightings and positions, helping the model be effective at recognising signs in many contexts. """)
    st.write("I partly used transfer learning and training my own layers. I used VGG16's CNN and added 1 pooling layer, 1 large hidden layer, and 1 output layer.")

    st.markdown("**Aim**")
    st.write("""My aim with this project was just to build an effective image classifier using transfer learning on a challenging dataset that I'm interested in.""")
    st.write("""I chose the ASL alphabet firstly because there were more readily available datasets.
             Secondly, because it would potentially be more of a challenge to classify than some alternatives.""")
    st.write("""I personally speak British Sign Language (BSL), as my parents are deaf. The BSL alphabet is a combination of one-handed signs and two-handed signs.
             While the ASL alphabet consists of only one-handed signs. My thinking was that with many similar looking signs (A&E, M&N, G&Z, etc.) it would be more of a challenge to classify.""")

    st.markdown("**Sign Language Processing (SLP)**")
    st.write("""SLP is a developing field of deep learning I have a personal interest in.
             It's complex and still evolving, but mainly combines both Natural Language Processing and Computer Vision.
             Classifying individual signs is very small step in the long process of SLP!
             Sign language is much more than just a sequence of signs, so in the SLP field lots of work and research is going on to capture the meaning expressed through body language, facial expression, sign momentum etc.
             """)

## subheaders
subheads = st.columns([4,10])
with subheads[0]:
    st.subheader('Try it yourself')
with subheads[1]:
    st.subheader('Example signs')

## split display into two panels
c= st.columns([4,1,1,1,1,1,1,1,1,1,1])
## combine 3 of the column object to get 3 rows of 10 tiles
cols = c[1:]+c[1:]+c[1:]

with c[0]:
    upload = st.file_uploader(label='Drag and drop an example or upload your own',type=['jpg','png'])

## loop through the tiles, images, and labels to display them in a grid
for tile, image, sign in zip(cols,images, signs):
    with tile:
        st.image(image)
        st.markdown(f"<h6 style='text-align: center; color: black;'>{sign} </h6>", unsafe_allow_html=True)

## function to process predictions
def max2(preds):
    '''
    Given the prediction array, return the index of two highest probabilities, so that their labels can be output later
    '''
    arr = preds.copy()
    idx1 = arr.argmax()
    arr[idx1] = 0

    idx2 = arr.argmax()

    return idx1, idx2

## is run when someone uplaods an image
if upload:
    ## preprocess image for model
    input = Image.open(upload)
    input = img_to_array(input)
    input = transform.resize(input, (220, 220, 3))
    input = preprocess_input(input)
    input = np.expand_dims(input, axis=0)
    ## send image to model
    pred = best_model.predict(input)
    ## output prediction
    idx1, idx2 = max2(pred[0])
    ## display prediction
    with c[0]:
        st.subheader('Prediction')
        st.markdown(f"<h4 style='font-weight: normal'>The result is: <b>{signs[idx1]} </b></h4>", unsafe_allow_html=True)
        st.image(Image.open(upload), width=100)
        st.markdown(f"But if not, the second most likely is... **{signs[idx2]}**")
