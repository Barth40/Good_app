import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.applications.densenet import preprocess_input

# Load the model
model = tf.keras.models.load_model('saved_model/bestest_weights.hdf5')

# Define the class labels
class_labels = ['MildDemented', 'ModeratedDemented', 'NonDemented', 'VeryMildDemented']

# Upload image file
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))
    
    # Display the image
    st.image(opencv_image, channels="RGB")

    # Preprocess the image
    resized = preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    # Generate prediction
    if st.button("Generate Prediction"):
        prediction = model.predict(img_reshape).argmax()
        predicted_label = class_labels[prediction]
        st.title("Predicted Label for the image is {}".format(predicted_label))
