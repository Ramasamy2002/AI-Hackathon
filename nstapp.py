import streamlit as st
import streamlit.components.v1 as stc
import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import cv2
import tempfile
from streamlit_webrtc import webrtc_streamer

# Define a list of style images with raw string literals for file paths
style_images = {
    "Picasso Vibes": r"C:\Users\91934\Downloads\style.jpg",
    "Van Gogh Dreams": r"C:\Users\91934\Downloads\style1.jpg",
    "Monet's Garden": r"C:\Users\91934\Downloads\style2.jpg",
    "Oil Painting": r"C:\Users\91934\Downloads\oilpainting.png",
}

def load_image(image_file):
    if isinstance(image_file, str):  # Check if it's a file path
        img = Image.open(image_file).convert("RGB")
    else:  # It's an UploadedFile object
        img = Image.open(image_file)
    
    # Save the image in a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
        img.save(temp_image.name)
        img_path = temp_image.name
    
    img = tf.io.read_file(img_path)
    os.remove(img_path)
    
    return img

def preprocess_image(img, target_size=(256, 256)):
    img = tf.image.decode_image(img, channels=3)        # making sure the image has 3 channels
    img = tf.image.convert_image_dtype(img, tf.float32) # making sure the image has dtype float 32
    img = tf.image.resize(img, target_size)             # resize the image
    img = img[tf.newaxis, :]
    return img

def main():
    st.title("Neural Style Transfer")
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    st.subheader("Upload Images")

    st.write("** Note: Select the content image or capture one from the webcam and choose a style **")

    content_image_option = st.radio("Select Content Image Option", ("Upload Content Image", "Capture from Webcam"))

    if content_image_option == "Upload Content Image":
        content_image = st.file_uploader("Upload Content Image", type=['png', 'jpeg', 'jpg'])
    else:
        content_image = st.camera_input("Take a picture")

    # Create a dropdown select widget for choosing a style
    selected_style = st.selectbox("Choose a Style", list(style_images.keys()))
    
    
    if st.button("Process"):
        if content_image is not None:
            content_image = load_image(content_image)
            content_image = preprocess_image(content_image)
            
            # Load the selected style image
            style_image_path = style_images[selected_style]
            style_image = load_image(style_image_path)
            style_image = preprocess_image(style_image)
            
            # Display the selected style image with a reduced size
            st.subheader("Style Image")
            st.image(style_image[0].numpy(), caption=f"Selected Style: {selected_style}", use_column_width=True)
            
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
            
            # Display the output image with a title
            st.subheader("Output Image")
            st.image((np.squeeze(stylized_image)), use_column_width=True)

if __name__ == '__main__':
    main()
