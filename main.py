import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer, DreamyImages
from helpful_functions import *
import numpy as np
import cv2
import tensorflow_hub as hub


# Loading the images
style_img = load_image("Data/starry_night.jpg")
content_img = load_image("Data/night_sky.jpg")

dreamer = DreamyImages()

# Transferring the style and content attributes to a noise image
nst = NeuralStyleTransfer(style_weight=1e-1, content_weight=1e3, tv_weight=1e-6)
gen_img = nst.transfer(tf.constant(style_img), tf.constant(content_img), epochs=100, image_size=448)

stylized_img = tensor_to_image(gen_img[0])

# Display the generated image by pretrained model
show_image(stylized_img)

frame = cv2.resize(frame, (300, 300))
frame = dreamer.generate_dream(frame, 100, 0.01)