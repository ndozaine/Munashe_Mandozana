import streamlit as st
from charset_normalizer import detect
from numpy import object_
import shutil
import cv2
import os
from PIL import Image
import numpy as np
from mailbox import ExternalClashError
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import io
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()

from keras.preprocessing.image import load_img
# load an image from file
from keras.preprocessing.image import img_to_array
# convert the image pixels to a numpy array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

## Function to detect object
def detect_Object():
  count = 0
  while count < 70:
    image = load_img('frames/frame%d.jpg' %count, target_size=(224, 224))
    image = img_to_array(image)
    # convert the image pixels to a numpy array
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # predict the probability across all output classes
    object = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(object)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    result=('%s (%.2f%%)' % (label[1], label[2] * 100))
    count = count + 1
    return result



## Function to save the uploaded file
def save_uploadedfile(uploaded_file):
    with open(os.path.join("uploadedVideos", uploaded_file.name), "wb") as f:
      f.write(uploaded_file.getbuffer())
      global filename
      filename = uploaded_file.name
      st.success("Saved File:{} to tempDir".format(uploaded_file.name))
      return filename
## Function to split video into frames
def generate_frames(video):
  vidcap = cv2.VideoCapture(video)
  success, image = vidcap.read()
  count = 0

  while success:
    cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  return

def main():
    """ """

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:black ;padding:10px">
    <h2 style="color:white;text-align:center;">Dhliwayo and Munashe Detecting Application_Lab</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a video...of not more than 2MB", type=["mp4"])
    temporary_location = False
    if uploaded_file is not None:
        filename = 'uploadedVideos/' + str(save_uploadedfile(uploaded_file))
        ## Split video into frames
        generate_frames(filename)
        ## Detect objects in frames
        #detect_Object()

    search_item = st.text_input('search object')
    if st.button("Search"):
        output = detect_Object()
        st.success(output)


if __name__ == '__main__':
    main()