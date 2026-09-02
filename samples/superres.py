#
# DirectML super-sesolution sample
# Based on the following model: https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016
#
import directml as dml
import numpy as np

import tensor_data
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "dog2.jpg"
tensor_data_path = "super_resolution_10_data"
batch_size = 1

# Get user image input path if any. If none, default to image_file_path value.
if (argument_count >= 2):
    image_file_path = sys.argv[1]

if (not os.path.exists(image_file_path)):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

# Image preprocessing
img = Image.open(image_file_path)
img = ImageOps.fit(img, (224, 224), method = 0, bleed = 0, centering = (0.5, 0.5))

img_ycbcr = img.convert('YCbCr')
img_y_0, img_cb, img_cr = img_ycbcr.split()
img_ndarray = np.asarray(img_y_0)

img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
img_5 = img_4.astype(np.float32) / 255.0

# Create an executing device and build a model
device = dml.Device(use_debug_layer=True)
graph = dml.Graph(device)

def load(file_name):
    return tensor_data.load(tensor_data_path + '/' + file_name)

input = graph.input([batch_size, 1, 224, 224])

# conv1
conv1_filter = graph.constant(load("conv1.weight.npy"), np.float32, sizes=[64, 1, 5, 5])
conv1_bias = graph.constant(load("conv1.bias.npy"), np.float32, sizes=[1,64,1,1])
conv1 = dml.convolution(input, conv1_filter, conv1_bias, start_padding = [2,2], end_padding = [2,2], fused_activation = dml.FusedActivation.relu())

# conv2
conv2_filter = graph.constant(load("conv2.weight.npy"), np.float32, sizes=[64,64,3,3])
conv2_bias = graph.constant(load("conv2.bias.npy"), np.float32, sizes=[1,64,1,1])
conv2 = dml.convolution(conv1, conv2_filter, conv2_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# conv3
conv3_filter = graph.constant(load("conv3.weight.npy"), np.float32, sizes=[32,64,3,3])
conv3_bias = graph.constant(load("conv3.bias.npy"), np.float32, sizes=[1,32,1,1])
conv3 = dml.convolution(conv2, conv3_filter, conv3_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

conv4_filter = graph.constant(load("conv4.weight.npy"), np.float32, sizes=[9, 32, 3, 3])
conv4_bias = graph.constant(load("conv4.bias.npy"), np.float32, sizes=[1,9,1,1])
conv4 = dml.convolution(conv3, conv4_filter, conv4_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# Compile the expression graph into a compiled operator
op = graph.compile([conv4])            # every filter and bias went up here

# Compute the result
output_tensor, = op({input: img_5})
output_tensor = np.reshape(output_tensor, [-1,1,3,3,224,224])
output_tensor = output_tensor.transpose((0, 1, 4, 2, 5, 3))
output_tensor = np.reshape(output_tensor, [-1,1,672,672])

for out in output_tensor[0]:
    img_out_y = Image.fromarray(np.uint8((out.squeeze() * 255.0).clip(0,255)), mode='L')
    final_img = Image.merge(
        "YCbCr", 
        [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]
    ).convert("RGB")
    final_img.show()