#
# DirectML FNS Candy sample
# Based on the following model: https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/FNSCandyStyleTransfer/UWP/cs/Assets/candy.onnx
#

import directml as dml
import numpy as np

import tensor_data
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "DefaultImage.jpg"
tensor_data_path = "candy_tensor_data"

# Get user image input path if any. If none, default to image_file_path value.
if (argument_count >= 2):
    image_file_path = sys.argv[1]

if (argument_count >= 3):
    tensor_data_path = sys.argv[2]

if (os.path.exists(image_file_path) == False):
    print("File not found at: " + str(image_file_path))
    sys.exit(1)

image = Image.open(image_file_path)

image = ImageOps.fit(image, (720, 720), method = 0,  bleed = 0, centering = (0.5, 0.5))

image_tensor = np.array(image, np.float32)

transposed_image = np.transpose(image_tensor, axes=[2,0,1])

# Create a GPU device and build a model graph.
device = dml.Device(use_debug_layer=True)
graph = dml.Graph(device)

def load(file_name):
    return tensor_data.load(tensor_data_path + '/' + file_name)

input = graph.input([1, 3, 720, 720])

# scalar1
scaler1_bias = [-103.93900299072266, -116.77899932861328, -123.68000030517578]
scaler1 = dml.value_scale_2d(input, scale=1.0, bias=scaler1_bias)

# pad3
pad3 = dml.padding(scaler1, padding_mode=dml.PaddingMode.REFLECTION,
                  start_padding=[0, 0, 40, 40], end_padding=[0, 0, 40, 40])

# conv4
conv4_filter = graph.constant(load("convolution_W.npy"), np.float32, sizes=[16,3,9,9])
conv4_bias = graph.constant(load("convolution_B.npy"), np.float32, sizes=[1,16,1,1])
conv4 = dml.convolution(pad3, conv4_filter, conv4_bias, start_padding = [4,4], end_padding = [4,4])

# instance_norm5
instance_norm5_scale = graph.input([1,16,1,1])
instance_norm5_bias = graph.constant(load("InstanceNormalization_B.npy"), np.float32, sizes=[1,16,1,1])
instance_norm5 = dml.mean_variance_normalization(conv4, instance_norm5_scale, instance_norm5_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv7
conv7_filter = graph.constant(load("convolution1_W.npy"), np.float32, sizes=[32,16,3,3])
conv7_bias = graph.constant(load("convolution1_B.npy"), np.float32, sizes=[1,32,1,1])
conv7 = dml.convolution(instance_norm5, conv7_filter, conv7_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# instance_norm8
instance_norm8_scale = graph.input([1,32,1,1])
instance_norm8_bias = graph.constant(load("InstanceNormalization_B1.npy"), np.float32, sizes=[1,32,1,1])
instance_norm8 = dml.mean_variance_normalization(conv7, instance_norm8_scale, instance_norm8_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv10
conv10_filter = graph.constant(load("convolution2_W.npy"), np.float32, sizes=[64,32,3,3])
conv10_bias = graph.constant(load("convolution2_B.npy"), np.float32, sizes=[1,64,1,1])
conv10 = dml.convolution(instance_norm8, conv10_filter, conv10_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# instance_norm11
instance_norm11_scale = graph.input([1,64,1,1])
instance_norm11_bias = graph.constant(load("InstanceNormalization_B2.npy"), np.float32, sizes=[1,64,1,1])
instance_norm11 = dml.mean_variance_normalization(conv10, instance_norm11_scale, instance_norm11_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv16
conv16_filter = graph.constant(load("convolution3_W.npy"), np.float32, sizes=[64,64,3,3])
conv16_bias = graph.constant(load("convolution3_B.npy"), np.float32, sizes=[1,64,1,1])
conv16 = dml.convolution(instance_norm11, conv16_filter, conv16_bias)

# instance_norm17
instance_norm17_scale = graph.input([1,64,1,1])
instance_norm17_bias = graph.constant(load("InstanceNormalization_B3.npy"), np.float32, sizes=[1,64,1,1])
instance_norm17 = dml.mean_variance_normalization(conv16, instance_norm17_scale, instance_norm17_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv19
conv19_filter = graph.constant(load("convolution4_W.npy"), np.float32, sizes=[64,64,3,3])
conv19_bias = graph.constant(load("convolution4_B.npy"), np.float32, sizes=[1,64,1,1])
conv19 = dml.convolution(instance_norm17, conv19_filter, conv19_bias)

# instance_norm15
instance_norm15_scale = graph.input([1,64,1,1])
instance_norm15_bias = graph.constant(load("InstanceNormalization_B4.npy"), np.float32, sizes=[1,64,1,1])
instance_norm15 = dml.mean_variance_normalization(conv19, instance_norm15_scale, instance_norm15_bias, axes=[0, 2, 3])

# crop21
crop21 = dml.slice(instance_norm11, input_window_offsets=[0,0,2,2], input_window_sizes=[1,64,196,196], input_window_strides=[1,1,1,1])

# add
add = dml.add(instance_norm15, crop21)

# conv26
conv26_filter = graph.constant(load("convolution5_W.npy"), np.float32, sizes=[64,64,3,3])
conv26_bias = graph.constant(load("convolution5_B.npy"), np.float32, sizes=[1,64,1,1])
conv26 = dml.convolution(add, conv26_filter, conv26_bias)

# instance_norm27
instance_norm27_scale = graph.input([1,64,1,1])
instance_norm27_bias = graph.constant(load("InstanceNormalization_B5.npy"), np.float32, sizes=[1,64,1,1])
instance_norm27 = dml.mean_variance_normalization(conv26, instance_norm27_scale, instance_norm27_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv29
conv29_filter = graph.constant(load("convolution6_W.npy"), np.float32, sizes=[64,64,3,3])
conv29_bias = graph.constant(load("convolution6_B.npy"), np.float32, sizes=[1,64,1,1])
conv29 = dml.convolution(instance_norm27, conv29_filter, conv29_bias)

# instance_norm25
instance_norm25_scale = graph.input([1,64,1,1])
instance_norm25_bias = graph.constant(load("InstanceNormalization_B6.npy"), np.float32, sizes=[1,64,1,1])
instance_norm25 = dml.mean_variance_normalization(conv29, instance_norm25_scale, instance_norm25_bias, axes=[0, 2, 3])

# crop21
crop31 = dml.slice(add, input_window_offsets=[0,0,2,2], input_window_sizes=[1,64,196-2-2,196-2-2], input_window_strides=[1,1,1,1])

# add1
add1 = dml.add(instance_norm25, crop31)

#conv36
conv36_filter = graph.constant(load("convolution7_W.npy"), np.float32, sizes=[64,64,3,3])
conv36_bias = graph.constant(load("convolution7_B.npy"), np.float32, sizes=[1,64,1,1])
conv36 = dml.convolution(add1, conv36_filter, conv36_bias)

# instance_norm37
instance_norm37_scale = graph.input([1,64,1,1])
instance_norm37_bias = graph.constant(load("InstanceNormalization_B7.npy"), np.float32, sizes=[1,64,1,1])
instance_norm37 = dml.mean_variance_normalization(conv36, instance_norm37_scale, instance_norm37_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv39
conv39_filter = graph.constant(load("convolution8_W.npy"), np.float32, sizes=[64,64,3,3])
conv39_bias = graph.constant(load("convolution8_B.npy"), np.float32, sizes=[1,64,1,1])
conv39 = dml.convolution(instance_norm37, conv39_filter, conv39_bias)

# instance_norm35
instance_norm35_scale = graph.input([1,64,1,1])
instance_norm35_bias = graph.constant(load("InstanceNormalization_B8.npy"), np.float32, sizes=[1,64,1,1])
instance_norm35 = dml.mean_variance_normalization(conv39, instance_norm35_scale, instance_norm35_bias, axes=[0, 2, 3])

# crop41
crop41 = dml.slice(add1, input_window_offsets=[0,0,2,2], input_window_sizes=[1, 64, 192-2-2, 192-2-2], input_window_strides=[1,1,1,1])

# add2
add2 = dml.add(instance_norm35, crop41)

# conv46
conv46_filter = graph.constant(load("convolution9_W.npy"), np.float32, sizes=[64,64,3,3])
conv46_bias = graph.constant(load("convolution9_B.npy"), np.float32, sizes=[1,64,1,1])
conv46 = dml.convolution(add2, conv46_filter, conv46_bias)

# instance_norm47
instance_norm47_scale = graph.input([1,64,1,1])
instance_norm47_bias = graph.constant(load("InstanceNormalization_B9.npy"), np.float32, sizes=[1,64,1,1])
instance_norm47 = dml.mean_variance_normalization(conv46, instance_norm47_scale, instance_norm47_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv49
conv49_filter = graph.constant(load("convolution10_W.npy"), np.float32, sizes=[64,64,3,3])
conv49_bias = graph.constant(load("convolution10_B.npy"), np.float32, sizes=[1,64,1,1])
conv49 = dml.convolution(instance_norm47, conv49_filter, conv49_bias)

# instance_norm45
instance_norm45_scale = graph.input([1,64,1,1])
instance_norm45_bias = graph.constant(load("InstanceNormalization_B10.npy"), np.float32, sizes=[1,64,1,1])
instance_norm45 = dml.mean_variance_normalization(conv49, instance_norm45_scale, instance_norm45_bias, axes=[0, 2, 3])

# crop51
crop51 = dml.slice(instance_norm35, input_window_offsets=[0,0,2,2], input_window_sizes=[1, 64, 188-2-2, 188-2-2], input_window_strides=[1,1,1,1])

# add3
add3 = dml.add(instance_norm45, crop51)

# conv56
conv56_filter = graph.constant(load("convolution11_W.npy"), np.float32, sizes=[64,64,3,3])
conv56_bias = graph.constant(load("convolution11_B.npy"), np.float32, sizes=[1,64,1,1])
conv56 = dml.convolution(add3, conv56_filter, conv56_bias)

# instance_norm57
instance_norm57_scale = graph.input([1,64,1,1])
instance_norm57_bias = graph.constant(load("InstanceNormalization_B11.npy"), np.float32, sizes=[1,64,1,1])
instance_norm57 = dml.mean_variance_normalization(conv56, instance_norm57_scale, instance_norm57_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv59
conv59_filter = graph.constant(load("convolution12_W.npy"), np.float32, sizes=[64,64,3,3])
conv59_bias = graph.constant(load("convolution12_B.npy"), np.float32, sizes=[1,64,1,1])
conv59 = dml.convolution(instance_norm57, conv59_filter, conv59_bias)

# instance_norm55
instance_norm55_scale = graph.input([1,64,1,1])
instance_norm55_bias = graph.constant(load("InstanceNormalization_B12.npy"), np.float32, sizes=[1,64,1,1])
instance_norm55 = dml.mean_variance_normalization(conv59, instance_norm55_scale, instance_norm55_bias, axes=[0, 2, 3])

# crop61
crop61 = dml.slice(add3, input_window_offsets=[0,0,2,2], input_window_sizes=[1, 64, 184-2-2, 184-2-2], input_window_strides=[1,1,1,1])

# add4
add4 = dml.add(instance_norm55, crop61)

# conv63
conv63_filter = graph.constant(load("convolution13_W.npy"), np.float32, sizes=[64,32,3,3])
conv63_bias = graph.constant(load("convolution13_B.npy"), np.float32, sizes=[1,32,1,1])
conv63 = dml.convolution(add4, conv63_filter, conv63_bias, mode = dml.ConvolutionMode.CROSS_CORRELATION, direction = dml.ConvolutionDirection.BACKWARD, strides = [2,2], start_padding = [0,0], end_padding = [0,0], output_sizes = [1,32,362,362])

# crop63
crop63 = dml.slice(conv63, input_window_offsets=[0,0,1,1], input_window_sizes=[1, 32, 362-1-1, 362-1-1], input_window_strides=[1,1,1,1])

# instance_norm65
instance_norm65_scale = graph.input([1,32,1,1])
instance_norm65_bias = graph.constant(load("InstanceNormalization_B13.npy"), np.float32, sizes=[1,32,1,1])
instance_norm65 = dml.mean_variance_normalization(crop63, instance_norm65_scale, instance_norm65_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv67
conv67_filter = graph.constant(load("convolution14_W.npy"), np.float32, sizes=[32,16,3,3])
conv67_bias = graph.constant(load("convolution14_B.npy"), np.float32, sizes=[1,16,1,1])
conv67 = dml.convolution(instance_norm65, conv67_filter, conv67_bias, mode = dml.ConvolutionMode.CROSS_CORRELATION, direction = dml.ConvolutionDirection.BACKWARD, strides = [2,2], start_padding = [0,0], end_padding = [0,0], output_sizes = [1,16, 722,722])

# crop67
crop67 = dml.slice(conv67, input_window_offsets=[0,0,1,1], input_window_sizes=[1, 16, 722-1-1, 722-1-1], input_window_strides=[1,1,1,1])

# instance_norm69
instance_norm69_scale = graph.input([1,16,1,1])
instance_norm69_bias = graph.constant(load("InstanceNormalization_B14.npy"), np.float32, sizes=[1,16,1,1])
instance_norm69 = dml.mean_variance_normalization(crop67, instance_norm69_scale, instance_norm69_bias, axes=[0, 2, 3], fused_activation=dml.FusedActivation.relu())

# conv71
conv71_filter = graph.constant(load("convolution15_W.npy"), np.float32, sizes=[3,16,9,9])
conv71_bias = graph.constant(load("convolution15_B.npy"), np.float32, sizes=[1,3,1,1])
conv71 = dml.convolution(instance_norm69, conv71_filter, conv71_bias, start_padding = [4,4], end_padding = [4,4], fused_activation = dml.FusedActivation.tanh())

# activation11
affine = dml.activation_linear(conv71, alpha=150, beta=0)

# deprocess_image_2_scaled
mul_b = graph.input([1,3,720,720], strides=[3,1,0,0])
image_2_scaled = dml.multiply(affine, mul_b)

# add_output
scale_b = graph.input([1,3,720,720], strides=[3,1,0,0])
output_image = dml.add(image_2_scaled, scale_b)

# Compile the expression graph into a compiled operator
op = graph.compile([output_image])

# What DirectML owns went up at compile; the rest is bound per dispatch.
feeds = {
    input: transposed_image,
    instance_norm5_scale: load("InstanceNormalization_scale.npy"),
    instance_norm8_scale: load("InstanceNormalization_scale1.npy"),
    instance_norm11_scale: load("InstanceNormalization_scale2.npy"),
    instance_norm17_scale: load("InstanceNormalization_scale3.npy"),
    instance_norm15_scale: load("InstanceNormalization_scale4.npy"),
    instance_norm27_scale: load("InstanceNormalization_scale5.npy"),
    instance_norm25_scale: load("InstanceNormalization_scale6.npy"),
    instance_norm37_scale: load("InstanceNormalization_scale7.npy"),
    instance_norm35_scale: load("InstanceNormalization_scale8.npy"),
    instance_norm47_scale: load("InstanceNormalization_scale9.npy"),
    instance_norm45_scale: load("InstanceNormalization_scale10.npy"),
    instance_norm57_scale: load("InstanceNormalization_scale11.npy"),
    instance_norm55_scale: load("InstanceNormalization_scale12.npy"),
    instance_norm65_scale: load("InstanceNormalization_scale13.npy"),
    instance_norm69_scale: load("InstanceNormalization_scale14.npy"),
    mul_b: load("Mul_B.npy"),
    scale_b: load("scale_B.npy"),
}

# Compute the result
output_tensor, = op(feeds)

transposed_output_image = np.transpose(output_tensor, axes = [0,2,3,1]).squeeze()

styled_image = Image.fromarray(np.uint8(transposed_output_image))

styled_image.show()
