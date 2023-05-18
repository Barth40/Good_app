import streamlit as st # type: ignore
import tensorflow as tf
import numpy as np
import cv2  # type: ignore
#from tensorflow.keras.preprocessing import image  # type: ignore
#from keras_preprocessing.image import image
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
#from tensorflow.keras.applications import ResNet50 # type: ignore
#from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input  # type: ignore
#from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.applications.densenet import preprocess_input as DenseNet_preprocess_input 
#from keras.applications.densenet import preprocess_input as DenseNet_preprocess_input 
#tf.keras.applications.vgg16.preprocess_input

#model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
model = tf.keras.models.load_model('saved_model/bestest_weights.hdf5', compile=False)
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")



#['CBB', 'CBSD', 'CGM', 'CMD', 'H']
#map_dict = {0: 'MildDemented', 1:'CBSD', 2:'CGM', 3: 'CMD', 4: 'Healthy'}
map_dict= {0:'  MildDemented', 1: 'ModeratedDemented',  2:'NonDemented', 3: 'VeryMildDemented'}

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224, 224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = DenseNet_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        #pred_array_scaled = np.expand_dims(img_reshape, axis=0) 
        #prediction = model.predict(pred_array_scaled).argmax()
        #below line for debugging
        #st.title("Predictions are ") #, model.predict(img_reshape))
        prediction = model.predict(img_reshape).argmax()
        #below line for debugging
        #st.title(prediction.astype(str))
        st.title("Predicted Label for the image is {}".format(map_dict [prediction]))