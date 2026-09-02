#
# DirectML MNIST sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/mnist/model/mnist-8.onnx
#

import directml as dml
import numpy as np

import tensor_data
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "mnist_image1.png"
tensor_data_path = "mnist_tensor_data"

if (argument_count >= 2):
    image_file_path = sys.argv[1]

if (argument_count >= 3):
    tensor_data_path = sys.argv[2]

if (os.path.exists(image_file_path) == False):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

# Opens image, converts to grayscale, resizes, and crops to the input size.
image = ImageOps.fit(ImageOps.grayscale(Image.open(image_file_path)), (28, 28), method = 0, bleed = 0, centering = (0.5, 0.5))

# Check top left pixel's color, and inverts image color if pixel is bright.
x, y = 0, 0
coordinate = x, y
if (image.getpixel(coordinate) >= 128):
    image = ImageOps.invert(image)

# Expand dimensions to 4d tensor format, and rescale values in range of 0.0 and 1.0.
img_array = np.array(image, np.float32)
ndarray_image = np.expand_dims(img_array, axis = (0, 1))
rescaled_image = ndarray_image / ndarray_image.max()

device = dml.Device()
graph = dml.Graph(device)

def load(file_name):
    return tensor_data.load(tensor_data_path + "/" + file_name)

def constant(sizes, file_name):
    """A weight, owned by DirectML and uploaded at compile."""
    return graph.constant(load(file_name), np.float32, sizes=sizes)

feeds = {}

def feed(sizes, file_name, strides=None):
    """An input bound at every dispatch, from a file next to the weights."""
    tensor = graph.input(sizes, strides=strides)
    feeds[tensor] = load(file_name)
    return tensor

input = graph.input([1, 1, 28, 28])

# convolution28
convolution28_weight = constant([8, 1, 5, 5], "Parameter5.npy")
convolution28_bias = graph.constant(np.zeros([1, 8, 1, 1], np.float32))
convolution28 = dml.convolution(input, convolution28_weight, convolution28_bias, strides = [1, 1], start_padding = [2, 2], end_padding = [2, 2])

# plus30
plus30_param6 = feed([1, 8, 28, 28], "Parameter6.npy", strides=[8, 1, 0, 0])
plus30 = dml.add(convolution28, plus30_param6)

# relu32
relu32 = dml.activation_relu(plus30)

# pooling66
pooling66 = dml.max_pooling(relu32, strides = [2, 2], window_sizes = [2, 2])

# convolution110
convolution110_weight = constant([16, 8, 5, 5], "Parameter87.npy")
convolution110_bias = graph.constant(np.zeros([1, 16, 1, 1], np.float32))
convolution110 = dml.convolution(pooling66.values, convolution110_weight, convolution110_bias, strides = [1, 1], start_padding = [2, 2], end_padding = [2, 2])

# plus112
plus112_param88 = feed([1, 16, 14, 14], "Parameter88.npy", strides=[16, 1, 0, 0])
plus112 = dml.add(convolution110, plus112_param88)

# relu114
relu114 = dml.activation_relu(plus112)

# pooling160
pooling160 = dml.max_pooling(relu114, strides = [3, 3], window_sizes = [3, 3])

# times212_reshape0
times212_reshape0 = dml.reinterpret(pooling160.values, [1, 1, 1, 256], [256, 256, 256, 1])

# times212_reshape1
times212_reshape1_param193 = constant([16, 4, 4, 10], "Parameter193.npy")
identity = dml.activation_identity(times212_reshape1_param193)
times212_reshape1 = dml.reinterpret(identity, [1, 1, 256, 10], [2560, 2560, 10, 1])

# times212
times212 = dml.gemm(times212_reshape0, times212_reshape1)

# plus214
plus214_param194 = constant([1, 1, 1, 10], "Parameter194.npy")
plus214 = dml.add(times212, plus214_param194)

softmax = dml.activation_softmax(plus214)
op = graph.compile([softmax])

output_tensor, = op({input: rescaled_image, **feeds})

number = np.argmax(output_tensor)
print("\nNumber is: {}".format(number, end=''))
print("Confidence: {:2.2f}%".format(np.amax(output_tensor) * 100))