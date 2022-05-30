from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import time
import numpy as np
import os

# Only use the below code if you have low resources.
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# For supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))


# Add text content to the app
st.title("Magenta Fast NST")
st.sidebar.header("TL;DR")
st.sidebar.info("""
    This is a fast style transfer app using the [Magenta](https://github.com/magenta/magenta/blob/main/magenta/models/arbitrary_image_stylization/README.md)
    library. The app uses the [Neural Style Transfer](https://arxiv.org/abs/1508.06576)
    model to transfer the style of one image to another.

    This app also uses the Tensorflow [esrgan-tf2](https://github.com/peteryuX/esrgan-tf2) image super resolution model only when GPU is available.
    """)
st.sidebar.info("Currently running on %s" % (tf.config.list_physical_devices()[0].device_type))


# load base models and initialize content and style images 
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

isr_model = None
if tf.config.list_physical_devices('GPU'):
    isr_model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")
    # necessary if you want to use the model in eager mode on CPU
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

content_img = None
style_img = None


# load content image and style image
content_img =  st.file_uploader("Upload content image", type=["jpg", "png"])
style_img = st.file_uploader("Upload style image", type=["jpg", "png"])

# convert streamlit image to eager tensor (1, 2048, 2048, 3)
if content_img is not None and style_img is not None:
    content_img = tf.io.decode_image(
        content_img.read(),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    style_img = tf.io.decode_image(
        style_img.read(),
        channels=3, dtype=tf.float32)[tf.newaxis, ...]
    # resize the image to the output size

def update_output_size():
    global content_img, style_img
    if content_img is not None and style_img is not None:
        output_size = st.session_state.imgSizeKey
        content_img = tf.image.resize(content_img, [output_size, output_size])
        #  resize the style image to the output size but do not exceed max size of 512
        output_size = min(output_size, 512)
        style_img = tf.image.resize(style_img, [output_size, output_size])
    st.write("Output image size: ", output_size)

# slider to change the output image size
st.sidebar.slider("Output image size", min_value=256, max_value=2048, key="imgSizeKey", value=1024, on_change=update_output_size)


# a button to run the model
if st.sidebar.button("Run model"):
    start = time.time()
    with st.spinner("Loading..."):
    # run the model and convert the output image to PIL image
        style_img = tf.nn.avg_pool(style_img, ksize=[3,3], strides=[1,1], padding='SAME')
        outputs = hub_module(tf.constant(content_img), tf.constant(style_img))
        outputs = outputs[0]
        stylized_image = outputs.numpy()
        stylized_image = Image.fromarray(np.uint8(stylized_image[0] * 255))
        # show the image
        st.image(stylized_image, caption="Stylized image", use_column_width=True)
    # Enable a button to convert output image to super resolution image only when GPU is available
    if tf.config.list_physical_devices('GPU'):
        if st.sidebar.button("Output super resolution image"):
            sr_image = isr_model(outputs) # Perform Super Resolution here
            # TODO: output image needs to be properly converted to PIL image
            sr_image = Image.fromarray(np.uint8(sr_image[0] * 255))
            st.image(sr_image, caption="Stylized super resolution image", use_column_width=True)

    end = time.time()
    st.write(f'Elapsed Time = {end - start} seconds')