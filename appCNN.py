import streamlit as st
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.models import Model

# Load word-to-index and index-to-word mappings
word_to_index = {}
with open("data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pickle.load(file)

index_to_word = {}
with open("data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pickle.load(file)

# Load the trained model
st.write("Loading the model...")
model = load_model('model_checkpoints/model_14.h5')

# Load ResNet50 for image feature extraction
resnet50_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))
resnet50_model = Model(resnet50_model.input, resnet50_model.layers[-2].output)

# Function to generate captions for an image
def predict_caption(photo):
    inp_text = "startseq"
    for i in range(80):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=80, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]
        inp_text += (' ' + word)
        if word == 'endseq':
            break
    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

# Preprocess image to match the input format of ResNet50
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Wrapper function to encode image (feature extraction)
def encode_image(img):
    img = preprocess_image(img)
    feature_vector = resnet50_model.predict(img)
    return feature_vector

# Streamlit interface
st.title("Image Caption Generator")
st.subheader("Upload an image to generate a caption")

# File uploader for image input
uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    img_data = plt.imread(uploaded_image)
    st.image(img_data, caption="Uploaded Image", use_column_width=True)
    
    # Run model and display the caption
    if st.button("Generate Caption"):
        st.write("Encoding the image...")
        photo = encode_image(uploaded_image).reshape((1, 2048))

        st.write("Running model to generate the caption...")
        caption = predict_caption(photo)
        st.write(f"**Generated Caption:** {caption}")
