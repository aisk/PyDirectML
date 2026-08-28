#
# DirectML MNIST sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/mnist/model/mnist-8.onnx
#

import directml as dml
import numpy as np
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "mnist_image1.png"
tensor_data_path = "mnist-8_tensor_data"

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

# Create a GPU device, and build a model graph.
device = dml.Device(True, True)
builder = dml.GraphBuilder(device)

data_type = dml.TensorDataType.FLOAT32
flags = dml.TensorFlags.OWNED_BY_DML

def load(file_name):
    return np.load(tensor_data_path + "/" + file_name)

input = dml.input_tensor(builder, 0, dml.TensorDesc(data_type, [1, 1, 28, 28]))

# convolution28
convolution28_weight = dml.input_tensor(builder, 1, dml.TensorDesc(data_type, flags, [8, 1, 5, 5]))
convolution28_bias = dml.input_tensor(builder, 2, dml.TensorDesc(data_type, flags, [1, 8, 1, 1]))
convolution28 = dml.convolution(input, convolution28_weight, convolution28_bias, strides = [1, 1], start_padding = [2, 2], end_padding = [2, 2])

# plus30
plus30_param6 = dml.input_tensor(builder, 3, dml.TensorDesc(data_type, [1, 8, 28, 28], [8, 1, 0, 0]))
plus30 = dml.add(convolution28, plus30_param6)

# relu32
relu32 = dml.activation_relu(plus30)

# pooling66
pooling66 = dml.max_pooling(relu32, strides = [2, 2], window_sizes = [2, 2])

# convolution110
convolution110_weight = dml.input_tensor(builder, 4, dml.TensorDesc(data_type, flags, [16, 8, 5, 5]))
convolution110_bias = dml.input_tensor(builder, 5, dml.TensorDesc(data_type, flags, [1, 16, 1, 1]))
convolution110 = dml.convolution(pooling66.values, convolution110_weight, convolution110_bias, strides = [1, 1], start_padding = [2, 2], end_padding = [2, 2])

# plus112
plus112_param88 = dml.input_tensor(builder, 6, dml.TensorDesc(data_type, [1, 16, 14, 14], [16, 1, 0, 0]))
plus112 = dml.add(convolution110, plus112_param88)

# relu114
relu114 = dml.activation_relu(plus112)

# pooling160
pooling160 = dml.max_pooling(relu114, strides = [3, 3], window_sizes = [3, 3])

# times212_reshape0
times212_reshape0 = dml.reinterpret(pooling160.values, dml.TensorDataType.FLOAT32, [1, 1, 1, 256], [256, 256, 256, 1])

# times212_reshape1
times212_reshape1_param193 = dml.input_tensor(builder, 7, dml.TensorDesc(data_type, flags, [16, 4, 4, 10]))
identity = dml.activation_identity(times212_reshape1_param193)
times212_reshape1 = dml.reinterpret(identity, dml.TensorDataType.FLOAT32, [1, 1, 256, 10], [2560, 2560, 10, 1])

# times212
times212 = dml.gemm(times212_reshape0, times212_reshape1)

# plus214
plus214_param194 = dml.input_tensor(builder, 8, dml.TensorDesc(data_type, flags, [1, 1, 1, 10]))
plus214 = dml.add(times212, plus214_param194)

softmax = dml.activation_softmax(plus214)
# Compile the expression graph into a compiled operator
op = builder.build(dml.ExecutionFlags.NONE, [softmax])

# The weights DirectML owns are read once here; the rest ride along with each
# dispatch.
op.initialize({
    convolution28_weight: load("Parameter5.npy"),
    convolution28_bias: np.zeros([1, 8, 1, 1], np.float32),
    convolution110_weight: load("Parameter87.npy"),
    convolution110_bias: np.zeros([1, 16, 1, 1], np.float32),
    times212_reshape1_param193: load("Parameter193.npy"),
    plus214_param194: load("Parameter194.npy"),
})

# Compute the result
output_tensor, = op({
    input: rescaled_image,
    plus30_param6: load("Parameter6.npy"),
    plus112_param88: load("Parameter88.npy"),
})

number = np.argmax(output_tensor)
print("\nNumber is: {}".format(number, end=''))
print("Confidence: {:2.2f}%".format(np.amax(output_tensor) * 100))