import tensorflow as tf
import matplotlib.pyplot as plt
from models import NeuralStyleTransfer
import numpy as np
import tensorflow_hub as hub

def load_image(image_path, image_size=448):

    # Read image from path
    image = tf.io.read_file(image_path)

    # Decode the jpeg to tensor
    image = tf.image.decode_jpeg(image, channels=3)

    # Cast the dtypes of image to float32
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image
    image = tf.image.resize(image, (image_size,image_size))

    # Add a dimension to make it look like a batch
    return image[tf.newaxis, :]

def tensor_to_image(tensor):
    # [0-1] to [0-255] scale
    tensor = tensor * 255
    # Tensor to numpy array for visualization
    tensor = np.array(tensor, dtype=np.uint8)
    # [1,image_size,image_size,3] to [image_size, image_size,3]
    if np.ndim(tensor) > 3:
        tensor = tensor[0]

    return tensor

def show_image(content_img, style_img, img):

    fig, ax = plt.subplots(1,3, figsize=(15, 15))

    ax[0].imshow(style_img[0])
    ax[0].axis('off')
    ax[1].imshow(content_img[0])
    ax[1].axis('off')
    if len(img.shape) > 3:
        img = img[0]

    ax[2].imshow(img)
    ax[2].axis('off')
    plt.show()