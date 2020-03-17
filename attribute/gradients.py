import numpy as np
import tensorflow as tf

class AttributionMask(object):
  """Base class for saliency masks"""
  def __init__(self, model):
    self.model = model

  def GetMask(self, x_value):
    raise NotImplementedError('A derived class should implemented GetMask()')

  def GetSmoothedMask(self, x_value, target_class_index, stdev_spread=.15, nsamples=25, magnitude=True, **kwargs):

    stdev = stdev_spread * (np.max(x_value) - np.min(x_value))

    total_gradients = np.zeros_like(x_value)
    for i in range(nsamples):
      noise = np.random.normal(0, stdev, x_value.shape)
      x_plus_noise = x_value + noise
      grad = self.GetMask(x_plus_noise, target_class_index, **kwargs)
      if magnitude:
        total_gradients += (grad * grad)
      else:
        total_gradients += grad

    return total_gradients / nsamples

class GradientAttribution(AttributionMask):
  def __init__(self, model):
    super().__init__(model)

  def GetMask(self, x_value, target_class_index):
    num_classes = self.model.output.shape[1]

    expected_output = tf.one_hot([target_class_index] * x_value.shape[0], num_classes)

    with tf.GradientTape() as tape:
        inputs = tf.cast(x_value, tf.float32)
        tape.watch(inputs)
        predictions = self.model(inputs)
        loss = tf.keras.losses.categorical_crossentropy(expected_output, predictions)

    return tape.gradient(loss, inputs)
