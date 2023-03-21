import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
import numpy as np


class NeuralStyleTransfer:

    def __init__(self,
                 style_weight,
                 content_weight,
                 tv_weight,
                 content_layers=None,
                 style_layers=None):

        # Initialize the pretrained model of VGG19
        self.base_model = VGG19(weights='imagenet', include_top=False)
        self.base_model.trainable = False

        # Weights of style and content
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.tv_loss_weight = tv_weight

        # Layer names for content and style
        if content_layers is None:
            self.content_layers = ["block5_conv2"]
        else:
            self.content_layers = content_layers
        if style_layers is None:
            self.style_layers = ['block1_conv1',
                                 'block2_conv1',
                                 'block3_conv1',
                                 'block4_conv1',
                                 'block5_conv1']
        else:
            self.style_layers = style_layers

        # Outputs of given style and content layers
        outputs = [self.base_model.get_layer(name).output for name in (self.style_layers + self.content_layers)]

        # Whole model with inputs and outputs
        self.whole_model = tf.keras.models.Model([self.base_model.input], outputs)

    def gram_matrix(self, input_tensor):

        result = tf.linalg.matmul(input_tensor, input_tensor, transpose_a=True)
        #result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

        input_shape = tf.shape(input_tensor)

        num_locations = float(np.prod(np.array(input_shape[1:3])))

        result = result / num_locations

        return result

    def calc_outputs(self, inputs):

        inputs = 255 * inputs

        preprocessed_input = preprocess_input(inputs)

        outputs = self.whole_model(preprocessed_input)

        style_outputs = outputs[:len(self.style_layers)]
        content_outputs = outputs[len(self.style_layers):]

        style_outputs = [self.gram_matrix(style_output) for style_output in style_outputs]

        return content_outputs, style_outputs

    @staticmethod
    def compute_loss(outputs, targets):
      
      # tf.math.add_n: returns the element-wise sum of a list of tensors
      return tf.add_n([tf.reduce_mean((outputs[key] - targets[key]) ** 2) for key in range(len(outputs))])
    
    @staticmethod
    def compute_tv_loss(output):

      return tf.reduce_sum(tf.image.total_variation(output))

    def calc_total_loss(self, output=None, content_outputs=None, content_targets=None, style_outputs=None, style_targets=None, content_only=False, style_only=False):

        if content_only:

          content_loss = self.compute_loss(content_outputs, content_targets)

          return content_loss

        elif style_only:

          style_loss = self.compute_loss(style_outputs, style_targets)

          return style_loss 

        else:

          style_loss = self.compute_loss(style_outputs, style_targets)
          style_loss *= (self.style_weight / len(self.style_layers))

          content_loss = self.compute_loss(content_outputs, content_targets)
          content_loss *= (self.content_weight / len(self.content_layers))

          tv_loss = self.compute_tv_loss(output)
          tv_loss *= tv_loss * self.tv_loss_weight

          return style_loss + content_loss + tv_loss, style_loss, content_loss, tv_loss

    def style_train(self, image, style_targets, which_layer, epochs):

      optimizer = tf.keras.optimizers.Adam(learning_rate =2e-2, beta_1=0.99, epsilon=0.1)

      images = []

      for epoch in range(epochs):

        with tf.GradientTape(persistent=True) as tape:

          _, style_outputs = self.calc_outputs(image)

          loss = self.calc_total_loss(image, style_outputs=style_outputs[which_layer], style_targets=style_targets[which_layer], style_only=True)

        if (epoch + 1) % 100 == 0:
              print(f"Epoch: {epoch+1}/{epochs}, Style Loss: {loss}")

        img_gradient = tape.gradient(loss, image)
        optimizer.apply_gradients([(img_gradient, image)])

        image.assign(tf.clip_by_value(image, 0.0, 1.0))

        if (epoch + 1) % 20 == 0:
          images.append(self.tensor_to_image(image))

      return images


    def content_train(self, image , content_targets, epochs):

      optimizer = tf.keras.optimizers.Adam(learning_rate =2e-2, beta_1=0.99, epsilon=0.1)

      images = []

      for epoch in range(epochs):

        with tf.GradientTape(persistent=True) as tape:

          content_outputs, _ = self.calc_outputs(image)
          loss = self.calc_total_loss(image, content_outputs, content_targets, content_only=True)

        if (epoch + 1) % 100 == 0:
              print(f"Epoch: {epoch+1}/{epochs}, Content Loss: {loss}")

        img_gradient = tape.gradient(loss, image)
        optimizer.apply_gradients([(img_gradient, image)])

        image.assign(tf.clip_by_value(image, 0.0, 1.0))

        if (epoch + 1) % 20 == 0:
          images.append(self.tensor_to_image(image))

      return images

    def train(self, image, style_targets=None, content_targets=None, epochs=1000, content_only=False, style_only=False):

        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-2, beta_1=0.99, epsilon=0.1)

        for epoch in range(epochs):

            with tf.GradientTape(persistent=True) as tape:
                content_outputs, style_outputs = self.calc_outputs(image)
                loss, style_loss, content_loss, tv_loss = self.calc_total_loss(image, content_outputs, style_outputs, style_targets, content_targets)

            """if (epoch + 1) % 100 == 0:
              print(f"Epoch: {epoch+1}/{epochs}, Total Loss: {loss}, Style Loss: {style_loss}, Content Loss: {content_loss}, Tv Loss: {tv_loss}")"""

            img_gradient = tape.gradient(loss, image)
            optimizer.apply_gradients([(img_gradient, image)])

            image.assign(tf.clip_by_value(image, 0.0, 1.0))

        return image

    def transfer(self, style_image, content_image, epochs=1000,image_size=448):

        _, style_targets = self.calc_outputs(style_image)
        content_targets, _ = self.calc_outputs(content_image)

        image = tf.random.uniform((1,image_size,image_size,3))
        image = tf.Variable(image)

        image = self.train(image, style_targets, content_targets, epochs)

        return image

    def content_transfer_only(self, content_image, epochs=1000, image_size=448):

      content_targets, _ = self.calc_outputs(content_image)

      image = tf.random.uniform((1, image_size, image_size, 3))
      image = tf.Variable(image)

      images = self.content_train(image, content_targets, epochs)

      return images

    def style_transfer_only(self, style_image, which_layer, epochs=1000, image_size=448):

      _, style_targets = self.calc_outputs(style_image)

      image = tf.random.uniform((1, image_size, image_size, 3))
      image = tf.Variable(image)

      images = self.style_train(image, style_targets, which_layer, epochs)

      return images

    def tensor_to_image(self, tensor):

      tensor = tensor * 255

      tensor = np.array(tensor, dtype=np.uint8)

      if np.ndim(tensor) > 3:
        tensor = tensor[0]
        
      return tensor


class DreamyImages:

    """
    DeepDream is the result of an experiment that aimed to visualize the internal patterns that are learned by
    neural network. Compute its gradient with respect to activations of a specific layer and then modify the
    image to increase the magnitude of such activations to in turn magnify the patterns.
    """

    def __init__(self, mixed_layer_names = None):

        self.base_model = InceptionV3(include_top=False, weights='imagenet')

        if mixed_layer_names is None:
            layer_names = ["mixed3","mixed5"]
        else:
            layer_names = mixed_layer_names

        outputs = [self.base_model.get_layer(layer_name).output for layer_name in layer_names]

        self.dreamer = tf.keras.Model(self.base_model.input, outputs)

    def _calculate_loss(self, image):
        """
        Calculate the loss between the extracted features and the image
        """

        image_batch = tf.expand_dims(image, 0)

        activations = self.dreamer(image_batch)

        if len(activations) == 1:

            activations = [activations]

        losses = []
        for activation in activations:
            loss = tf.math.reduce_mean(activation)
            losses.append(loss)

        total_loss = tf.math.reduce_sum(losses)

        return total_loss

    @tf.function
    def perform_gradient_ascent(self, image, steps, step_size):
        """
        A method to perform gradient ascent

        image = image + d(Loss) / d(Image) * step_size
        """

        loss = tf.constant(0.0)

        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(image)
                loss = self._calculate_loss(image)

            gradient = tape.gradient(loss, image)

            gradient = gradient / (tf.math.reduce_std(gradient) + 1e-8)

            image = image + gradient * step_size
            # Clips tensor values to a specified min and max.
            image = tf.clip_by_value(image, -1, 1)

        return loss, image

    def generate_dream(self, image, steps, step_size):
        """
        Generates dreamy images
        """

        image = preprocess_input(image)
        image = tf.convert_to_tensor(image)
        step_size = tf.convert_to_tensor(step_size)
        step_size = tf.constant(step_size)
        steps = tf.constant(steps)

        loss, image = self.perform_gradient_ascent(image, steps, step_size)

        image = 255 * (image + 1.0) / 2
        image = tf.cast(image, tf.uint8)

        return np.array(image)








