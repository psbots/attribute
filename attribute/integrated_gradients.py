import tensorflow as tf
import numpy as np
from .gradients import GradientAttribution

class IntegratedGradients(GradientAttribution):

  def GetMask(self, x_value, feed_dict={}, x_baseline=None, x_steps=25):

    if x_baseline is None:
      x_baseline = np.zeros_like(x_value)

    assert x_baseline.shape == x_value.shape

    x_diff = x_value - x_baseline

    total_gradients = np.zeros_like(x_value)

    for alpha in np.linspace(0, 1, x_steps):
      x_step = x_baseline + alpha * x_diff

      total_gradients += super().GetMask(x_step, feed_dict)

    return total_gradients * x_diff / x_steps
