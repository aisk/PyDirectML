#
# DirectML SqueezeNet sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/squeezenet/model/squeezenet1.1-7.onnx
#

import directml as dml
import numpy as np

import tensor_data
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "DefaultImage.jpg"
tensor_data_path = "squeezenet_tensor_data"

if (argument_count >= 2):
    image_file_path = sys.argv[1]

if (argument_count >= 3):
    tensor_data_path = sys.argv[2]

if (os.path.exists(image_file_path) == False):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

# Opens image, converts to RGB (in case grayscale or contains an alpha channel), resizes, and crops to the input size.
image = ImageOps.fit(Image.open(image_file_path).convert("RGB"), (224, 224), method = 0, bleed = 0, centering = (0.5, 0.5))

# Transposes image array from (H x W x C) to (C x H x W) and rescales its value to between 0 and 1.
ndarray_image = np.transpose(np.array(image, np.float32), axes = [2, 0, 1])
rescaled_image = ndarray_image / ndarray_image.max()

# Normalizes the rescaled image values using the model training statistics.
mean = np.array([[[0.485]],[[ 0.456]],[[0.406]]])
standard_deviation = np.array([[[0.229]],[[0.224]],[[0.225]]])
processed_image = (rescaled_image - mean) / standard_deviation

feeds = {}        # tensors bound per dispatch

def append_input_tensor(graph: dml.Graph, sizes: list, owned: bool, file_name: str):
    if file_name == "":
        array = np.zeros(sizes, np.float32)
    else:
        array = tensor_data.load(tensor_data_path + "/" + file_name)
    if owned:
        # A weight: the graph records it now and uploads it at compile.
        return graph.constant(array, np.float32, sizes=sizes)
    tensor = graph.input(sizes)
    feeds[tensor] = array
    return tensor

# Create a GPU device, and build a model graph.
device = dml.Device(use_debug_layer=True)
graph = dml.Graph(device)

input = graph.input([1, 3, 224, 224])


# squeezenet0_conv0_fwd
squeezenet0_conv0_weight = append_input_tensor(graph, [64,3,3,3], True, "squeezenet0_conv0_weight.npy")
squeezenet0_conv0_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv0_bias.npy")
squeezenet0_conv0_fwd = dml.convolution(input, squeezenet0_conv0_weight, squeezenet0_conv0_bias, strides = [2,2], fused_activation = dml.FusedActivation.relu())

# squeezenet0_pool0_fwd
squeezenet0_pool0_fwd = dml.max_pooling(squeezenet0_conv0_fwd, window_sizes = [3,3], strides = [2,2])

# squeezenet0_conv1_fwd
squeezenet0_conv1_weight = append_input_tensor(graph, [16,64,1,1], True, "squeezenet0_conv1_weight.npy")
squeezenet0_conv1_bias = append_input_tensor(graph, [1,16,1,1], True, "squeezenet0_conv1_bias.npy")
squeezenet0_conv1_fwd = dml.convolution(squeezenet0_pool0_fwd.values, squeezenet0_conv1_weight, squeezenet0_conv1_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv2_fwd
squeezenet0_conv2_weight = append_input_tensor(graph, [64,16,1,1], True, "squeezenet0_conv2_weight.npy")
squeezenet0_conv2_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv2_bias.npy")
squeezenet0_conv2_fwd = dml.convolution(squeezenet0_conv1_fwd, squeezenet0_conv2_weight, squeezenet0_conv2_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv3_fwd
squeezenet0_conv3_weight = append_input_tensor(graph, [64,16,3,3], True, "squeezenet0_conv3_weight.npy")
squeezenet0_conv3_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv3_bias.npy")
squeezenet0_conv3_fwd = dml.convolution(squeezenet0_conv1_fwd, squeezenet0_conv3_weight, squeezenet0_conv3_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat0
squeezenet0_concat0 = dml.join([squeezenet0_conv2_fwd, squeezenet0_conv3_fwd], axis=1)

# squeezenet0_conv4_fwd
squeezenet0_conv4_weight = append_input_tensor(graph, [16,128,1,1], True, "squeezenet0_conv4_weight.npy")
squeezenet0_conv4_bias = append_input_tensor(graph, [1,16,1,1], True, "squeezenet0_conv4_bias.npy")
squeezenet0_conv4_fwd = dml.convolution(squeezenet0_concat0, squeezenet0_conv4_weight, squeezenet0_conv4_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv5_fwd
squeezenet0_conv5_weight = append_input_tensor(graph, [64,16,1,1], True, "squeezenet0_conv5_weight.npy")
squeezenet0_conv5_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv5_bias.npy")
squeezenet0_conv5_fwd = dml.convolution(squeezenet0_conv4_fwd, squeezenet0_conv5_weight, squeezenet0_conv5_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv6_fwd
squeezenet0_conv6_weight = append_input_tensor(graph, [64,16,3,3], True, "squeezenet0_conv6_weight.npy")
squeezenet0_conv6_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv6_bias.npy")
squeezenet0_conv6_fwd = dml.convolution(squeezenet0_conv4_fwd, squeezenet0_conv6_weight, squeezenet0_conv6_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat1
squeezenet0_concat1 = dml.join([squeezenet0_conv5_fwd, squeezenet0_conv6_fwd], axis=1)

# squeezenet0_pool0_fwd
squeezenet0_pool1_fwd = dml.max_pooling(squeezenet0_concat1, window_sizes = [3,3], strides = [2,2])

# squeezenet0_conv7_fwd
squeezenet0_conv7_weight = append_input_tensor(graph, [32,128,1,1], True, "squeezenet0_conv7_weight.npy")
squeezenet0_conv7_bias = append_input_tensor(graph, [1,32,1,1], True, "squeezenet0_conv7_bias.npy")
squeezenet0_conv7_fwd = dml.convolution(squeezenet0_pool1_fwd.values, squeezenet0_conv7_weight, squeezenet0_conv7_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv8_fwd
squeezenet0_conv8_weight = append_input_tensor(graph, [128,32,1,1], True, "squeezenet0_conv8_weight.npy")
squeezenet0_conv8_bias = append_input_tensor(graph, [1,128,1,1], True, "squeezenet0_conv8_bias.npy")
squeezenet0_conv8_fwd = dml.convolution(squeezenet0_conv7_fwd, squeezenet0_conv8_weight, squeezenet0_conv8_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv9_fwd
squeezenet0_conv9_weight = append_input_tensor(graph, [128,32,3,3], True, "squeezenet0_conv9_weight.npy")
squeezenet0_conv9_bias = append_input_tensor(graph, [1,128,1,1], True, "squeezenet0_conv9_bias.npy")
squeezenet0_conv9_fwd = dml.convolution(squeezenet0_conv7_fwd, squeezenet0_conv9_weight, squeezenet0_conv9_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat2
squeezenet0_concat2 = dml.join([squeezenet0_conv8_fwd, squeezenet0_conv9_fwd], axis=1)

# squeezenet0_conv10_fwd
squeezenet0_conv10_weight = append_input_tensor(graph, [32,256,1,1], True, "squeezenet0_conv10_weight.npy")
squeezenet0_conv10_bias = append_input_tensor(graph, [1,32,1,1], True, "squeezenet0_conv10_bias.npy")
squeezenet0_conv10_fwd = dml.convolution(squeezenet0_concat2, squeezenet0_conv10_weight, squeezenet0_conv10_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv11_fwd
squeezenet0_conv11_weight = append_input_tensor(graph, [128,32,1,1], True, "squeezenet0_conv11_weight.npy")
squeezenet0_conv11_bias = append_input_tensor(graph, [1,128,1,1], True, "squeezenet0_conv11_bias.npy")
squeezenet0_conv11_fwd = dml.convolution(squeezenet0_conv10_fwd, squeezenet0_conv11_weight, squeezenet0_conv11_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv12_fwd
squeezenet0_conv12_weight = append_input_tensor(graph, [128,32,3,3], True, "squeezenet0_conv12_weight.npy")
squeezenet0_conv12_bias = append_input_tensor(graph, [1,128,1,1], True, "squeezenet0_conv12_bias.npy")
squeezenet0_conv12_fwd = dml.convolution(squeezenet0_conv10_fwd, squeezenet0_conv12_weight, squeezenet0_conv12_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat3
squeezenet0_concat3 = dml.join([squeezenet0_conv11_fwd, squeezenet0_conv12_fwd], axis=1)

# squeezenet0_pool2_fwd
squeezenet0_pool2_fwd = dml.max_pooling(squeezenet0_concat3, window_sizes = [3,3], strides = [2,2])

# squeezenet0_conv13_fwd
squeezenet0_conv13_weight = append_input_tensor(graph, [48,256,1,1], True, "squeezenet0_conv13_weight.npy")
squeezenet0_conv13_bias = append_input_tensor(graph, [1,48,1,1], True, "squeezenet0_conv13_bias.npy")
squeezenet0_conv13_fwd = dml.convolution(squeezenet0_pool2_fwd.values, squeezenet0_conv13_weight, squeezenet0_conv13_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv14_fwd
squeezenet0_conv14_weight = append_input_tensor(graph, [192,48,1,1], True, "squeezenet0_conv14_weight.npy")
squeezenet0_conv14_bias = append_input_tensor(graph, [1,192,1,1], True, "squeezenet0_conv14_bias.npy")
squeezenet0_conv14_fwd = dml.convolution(squeezenet0_conv13_fwd, squeezenet0_conv14_weight, squeezenet0_conv14_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv15_fwd
squeezenet0_conv15_weight = append_input_tensor(graph, [192,48,3,3], True, "squeezenet0_conv15_weight.npy")
squeezenet0_conv15_bias = append_input_tensor(graph, [1,192,1,1], True, "squeezenet0_conv15_bias.npy")
squeezenet0_conv15_fwd = dml.convolution(squeezenet0_conv13_fwd, squeezenet0_conv15_weight, squeezenet0_conv15_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat4
squeezenet0_concat4 = dml.join([squeezenet0_conv14_fwd, squeezenet0_conv15_fwd], axis=1)

# squeezenet0_conv16_fwd
squeezenet0_conv16_weight = append_input_tensor(graph, [48,384,1,1], True, "squeezenet0_conv16_weight.npy")
squeezenet0_conv16_bias = append_input_tensor(graph, [1,48,1,1], True, "squeezenet0_conv16_bias.npy")
squeezenet0_conv16_fwd = dml.convolution(squeezenet0_concat4, squeezenet0_conv16_weight, squeezenet0_conv16_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv17_fwd
squeezenet0_conv17_weight = append_input_tensor(graph, [192,48,1,1], True, "squeezenet0_conv17_weight.npy")
squeezenet0_conv17_bias = append_input_tensor(graph, [1,192,1,1], True, "squeezenet0_conv17_bias.npy")
squeezenet0_conv17_fwd = dml.convolution(squeezenet0_conv16_fwd, squeezenet0_conv17_weight, squeezenet0_conv17_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv18_fwd
squeezenet0_conv18_weight = append_input_tensor(graph, [192,48,3,3], True, "squeezenet0_conv18_weight.npy")
squeezenet0_conv18_bias = append_input_tensor(graph, [1,192,1,1], True, "squeezenet0_conv18_bias.npy")
squeezenet0_conv18_fwd = dml.convolution(squeezenet0_conv16_fwd, squeezenet0_conv18_weight, squeezenet0_conv18_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat5
squeezenet0_concat5 = dml.join([squeezenet0_conv17_fwd, squeezenet0_conv18_fwd], axis=1)

# squeezenet0_conv19_fwd
squeezenet0_conv19_weight = append_input_tensor(graph, [64,384,1,1], True, "squeezenet0_conv19_weight.npy")
squeezenet0_conv19_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv19_bias.npy")
squeezenet0_conv19_fwd = dml.convolution(squeezenet0_concat5, squeezenet0_conv19_weight, squeezenet0_conv19_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv20_fwd
squeezenet0_conv20_weight = append_input_tensor(graph, [256,64,1,1], True, "squeezenet0_conv20_weight.npy")
squeezenet0_conv20_bias = append_input_tensor(graph, [1,256,1,1], True, "squeezenet0_conv20_bias.npy")
squeezenet0_conv20_fwd = dml.convolution(squeezenet0_conv19_fwd, squeezenet0_conv20_weight, squeezenet0_conv20_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv21_fwd
squeezenet0_conv21_weight = append_input_tensor(graph, [256,64,3,3], True, "squeezenet0_conv21_weight.npy")
squeezenet0_conv21_bias = append_input_tensor(graph, [1,256,1,1], True, "squeezenet0_conv21_bias.npy")
squeezenet0_conv21_fwd = dml.convolution(squeezenet0_conv19_fwd, squeezenet0_conv21_weight, squeezenet0_conv21_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat6
squeezenet0_concat6 = dml.join([squeezenet0_conv20_fwd, squeezenet0_conv21_fwd], axis=1)

# squeezenet0_conv22_fwd
squeezenet0_conv22_weight = append_input_tensor(graph, [64,512,1,1], True, "squeezenet0_conv22_weight.npy")
squeezenet0_conv22_bias = append_input_tensor(graph, [1,64,1,1], True, "squeezenet0_conv22_bias.npy")
squeezenet0_conv22_fwd = dml.convolution(squeezenet0_concat6, squeezenet0_conv22_weight, squeezenet0_conv22_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv23_fwd
squeezenet0_conv23_weight = append_input_tensor(graph, [256,64,1,1], True, "squeezenet0_conv23_weight.npy")
squeezenet0_conv23_bias = append_input_tensor(graph, [1,256,1,1], True, "squeezenet0_conv23_bias.npy")
squeezenet0_conv23_fwd = dml.convolution(squeezenet0_conv22_fwd, squeezenet0_conv23_weight, squeezenet0_conv23_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_conv24_fwd
squeezenet0_conv24_weight = append_input_tensor(graph, [256,64,3,3], True, "squeezenet0_conv24_weight.npy")
squeezenet0_conv24_bias = append_input_tensor(graph, [1,256,1,1], True, "squeezenet0_conv24_bias.npy")
squeezenet0_conv24_fwd = dml.convolution(squeezenet0_conv22_fwd, squeezenet0_conv24_weight, squeezenet0_conv24_bias, start_padding = [1,1], end_padding = [1,1], fused_activation = dml.FusedActivation.relu())

# squeezenet0_concat7
squeezenet0_concat7 = dml.join([squeezenet0_conv23_fwd, squeezenet0_conv24_fwd], axis=1)

# squeezenet0_dropout0_fwd
squeezenet0_dropout0_fwd = dml.activation_identity(squeezenet0_concat7)

# squeezenet0_conv25_fwd
squeezenet0_conv25_weight = append_input_tensor(graph, [1000,512,1,1], True, "squeezenet0_conv25_weight.npy")
squeezenet0_conv25_bias = append_input_tensor(graph, [1,1000,1,1], True, "squeezenet0_conv25_bias.npy")
squeezenet0_conv25_fwd = dml.convolution(squeezenet0_dropout0_fwd, squeezenet0_conv25_weight, squeezenet0_conv25_bias, fused_activation = dml.FusedActivation.relu())

# squeezenet0_pool3_fwd
squeezenet0_pool3_fwd = dml.average_pooling(squeezenet0_conv25_fwd, strides=[13,13], window_sizes=[13,13], start_padding=[0,0], end_padding=[0,0])

# squeezenet0_flatten0_reshape0
squeezenet0_flatten0_reshape0 =dml.reinterpret(squeezenet0_pool3_fwd, [1,1,1,1000], [1000,1000,1000,1])

# softmax
soft_max = dml.activation_softmax(squeezenet0_flatten0_reshape0)

# Compile the expression graph into a compiled operator
op = graph.compile([soft_max])

# Compute the result
output_tensor, = op({input: processed_image, **feeds})

# Opens text file of categories to collect the correct image category
label_file = open("imagenet_categories.txt","r")
label_lines = label_file.readlines()
prediction_index = np.argmax(output_tensor)

# Print the program confidence and the category from locally stored ImageNet text file
print("\nCategory: {}".format(label_lines[prediction_index], end=''))
print("Confidence: {:2.2f}%".format(np.amax(output_tensor) * 100))
