#
# DirectML MoblieNet sample
# Based on the following model: https://github.com/onnx/models/blob/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx
#

import directml as dml
import numpy as np

import tensor_data
from PIL import Image, ImageOps
import sys
import os

argument_count = len(sys.argv)

image_file_path = "DefaultImage.jpg"
tensor_data_path = "mobilenet_tensor_data"

if (argument_count == 2):
    image_file_path = sys.argv[1]

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

input = graph.input([1,3,224,224])


# conv1 
conv1_filter = constant([32,3,3,3], "mobilenetv20_features_conv0_weight.npy")
conv1_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv1 = dml.convolution(input, conv1_filter, conv1_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1])

# batch_norm1
batch_norm1_mean = feed([1,32,1,1], "mobilenetv20_features_batchnorm0_running_mean.npy")
batch_norm1_variance = feed([1,32,1,1], "mobilenetv20_features_batchnorm0_running_var.npy")
batch_norm1_scale = feed([1,32,1,1], "mobilenetv20_features_batchnorm0_gamma.npy")
batch_norm1_bias = feed([1,32,1,1], "mobilenetv20_features_batchnorm0_beta.npy")
batch_norm1 = dml.batch_normalization(conv1, batch_norm1_mean, batch_norm1_variance, batch_norm1_scale, batch_norm1_bias, fused_activation=dml.FusedActivation.relu())

# conv2
conv2_filter = constant([32,32,1,1], "mobilenetv20_features_linearbottleneck0_conv0_weight.npy")
conv2_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv2 = dml.convolution(batch_norm1, conv2_filter, conv2_bias)

# batch_norm2
batch_norm2_mean = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm0_running_mean.npy")
batch_norm2_variance = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm0_running_var.npy")
batch_norm2_scale = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm0_gamma.npy")
batch_norm2_bias = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm0_beta.npy")
batch_norm2 = dml.batch_normalization(conv2, batch_norm2_mean, batch_norm2_variance, batch_norm2_scale, batch_norm2_bias, fused_activation=dml.FusedActivation.relu())

# conv3
conv3_filter = constant([32,1,3,3], "mobilenetv20_features_linearbottleneck0_conv1_weight.npy")
conv3_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv3 = dml.convolution(batch_norm2, conv3_filter, conv3_bias, start_padding = [1,1], end_padding = [1,1], group_count = 32)

# batch_norm3
batch_norm3_mean = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm1_running_mean.npy")
batch_norm3_variance = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm1_running_var.npy")
batch_norm3_scale = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm1_gamma.npy")
batch_norm3_bias = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm1_beta.npy")
batch_norm3 = dml.batch_normalization(conv3, batch_norm3_mean, batch_norm3_variance, batch_norm3_scale, batch_norm3_bias, fused_activation=dml.FusedActivation.relu())

# conv4
conv4_filter = constant([16,32,1,1], "mobilenetv20_features_linearbottleneck0_conv2_weight.npy")
conv4_bias = graph.constant(np.zeros([1,16,1,1], np.float32))
conv4 = dml.convolution(batch_norm3, conv4_filter, conv4_bias)

# batch_norm4
batch_norm4_mean = feed([1,16,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm2_running_mean.npy")
batch_norm4_variance = feed([1,16,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm2_running_var.npy")
batch_norm4_scale = feed([1,16,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm2_gamma.npy")
batch_norm4_bias = feed([1,16,1,1], "mobilenetv20_features_linearbottleneck0_batchnorm2_beta.npy")
batch_norm4 = dml.batch_normalization(conv4, batch_norm4_mean, batch_norm4_variance, batch_norm4_scale, batch_norm4_bias)

# conv5
conv5_filter = constant([96,16,1,1], "mobilenetv20_features_linearbottleneck1_conv0_weight.npy")
conv5_bias = graph.constant(np.zeros([1,96,1,1], np.float32))
conv5 = dml.convolution(batch_norm4, conv5_filter, conv5_bias)

# batch_norm5
batch_norm5_mean = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm0_running_mean.npy")
batch_norm5_variance = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm0_running_var.npy")
batch_norm5_scale = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm0_gamma.npy")
batch_norm5_bias = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm0_beta.npy")
batch_norm5 = dml.batch_normalization(conv5, batch_norm5_mean, batch_norm5_variance, batch_norm5_scale, batch_norm5_bias, fused_activation=dml.FusedActivation.relu())

# conv6
conv6_filter = constant([96,1,3,3], "mobilenetv20_features_linearbottleneck1_conv1_weight.npy")
conv6_bias = graph.constant(np.zeros([1,96,1,1], np.float32))
conv6 = dml.convolution(batch_norm5, conv6_filter, conv6_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 96)

# batch_norm6
batch_norm6_mean = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm1_running_mean.npy")
batch_norm6_variance = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm1_running_var.npy")
batch_norm6_scale = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm1_gamma.npy")
batch_norm6_bias = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm1_beta.npy")
batch_norm6 = dml.batch_normalization(conv6, batch_norm6_mean, batch_norm6_variance, batch_norm6_scale, batch_norm6_bias, fused_activation=dml.FusedActivation.relu())

# conv7
conv7_filter = constant([24,96,1,1], "mobilenetv20_features_linearbottleneck1_conv2_weight.npy")
conv7_bias = graph.constant(np.zeros([1,24,1,1], np.float32))
conv7 = dml.convolution(batch_norm6, conv7_filter, conv7_bias)

# batch_norm7
batch_norm7_mean = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm2_running_mean.npy")
batch_norm7_variance = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm2_running_var.npy")
batch_norm7_scale = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm2_gamma.npy")
batch_norm7_bias = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck1_batchnorm2_beta.npy")
batch_norm7 = dml.batch_normalization(conv7, batch_norm7_mean, batch_norm7_variance, batch_norm7_scale, batch_norm7_bias)

# conv8
conv8_filter = constant([144,24,1,1], "mobilenetv20_features_linearbottleneck2_conv0_weight.npy")
conv8_bias = graph.constant(np.zeros([1,144,1,1], np.float32))
conv8 = dml.convolution(batch_norm7, conv8_filter, conv8_bias)

# batch_norm8
batch_norm8_mean = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm0_running_mean.npy")
batch_norm8_variance = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm0_running_var.npy")
batch_norm8_scale = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm0_gamma.npy")
batch_norm8_bias = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm0_beta.npy")
batch_norm8 = dml.batch_normalization(conv8, batch_norm8_mean, batch_norm8_variance, batch_norm8_scale, batch_norm8_bias, fused_activation=dml.FusedActivation.relu())

# conv9
conv9_filter = constant([144,1,3,3], "mobilenetv20_features_linearbottleneck2_conv1_weight.npy")
conv9_bias = graph.constant(np.zeros([1,144,1,1], np.float32))
conv9 = dml.convolution(batch_norm8, conv9_filter, conv9_bias, start_padding = [1,1], end_padding = [1,1], group_count = 144)

# batch_norm9
batch_norm9_mean = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm1_running_mean.npy")
batch_norm9_variance = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm1_running_var.npy")
batch_norm9_scale = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm1_gamma.npy")
batch_norm9_bias = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm1_beta.npy")
batch_norm9 = dml.batch_normalization(conv9, batch_norm9_mean, batch_norm9_variance, batch_norm9_scale, batch_norm9_bias, fused_activation=dml.FusedActivation.relu())

# conv10
conv10_filter = constant([24,144,1,1], "mobilenetv20_features_linearbottleneck2_conv2_weight.npy")
conv10_bias = graph.constant(np.zeros([1,24,1,1], np.float32))
conv10 = dml.convolution(batch_norm9, conv10_filter, conv10_bias)

# batch_norm10
batch_norm10_mean = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm2_running_mean.npy")
batch_norm10_variance = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm2_running_var.npy")
batch_norm10_scale = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm2_gamma.npy")
batch_norm10_bias = feed([1,24,1,1], "mobilenetv20_features_linearbottleneck2_batchnorm2_beta.npy")
batch_norm10 = dml.batch_normalization(conv10, batch_norm10_mean, batch_norm10_variance, batch_norm10_scale, batch_norm10_bias)

# add1
add1 = dml.add(batch_norm7, batch_norm10)

# conv11
conv11_filter = constant([144,24,1,1], "mobilenetv20_features_linearbottleneck3_conv0_weight.npy")
conv11_bias = graph.constant(np.zeros([1,144,1,1], np.float32))
conv11 = dml.convolution(add1, conv11_filter, conv11_bias)

# batch_norm11
batch_norm11_mean = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm0_running_mean.npy")
batch_norm11_variance = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm0_running_var.npy")
batch_norm11_scale = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm0_gamma.npy")
batch_norm11_bias = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm0_beta.npy")
batch_norm11 = dml.batch_normalization(conv11, batch_norm11_mean, batch_norm11_variance, batch_norm11_scale, batch_norm11_bias, fused_activation=dml.FusedActivation.relu())

# conv12
conv12_filter = constant([144,1,3,3], "mobilenetv20_features_linearbottleneck3_conv1_weight.npy")
conv12_bias = graph.constant(np.zeros([1,144,1,1], np.float32))
conv12 = dml.convolution(batch_norm11, conv12_filter, conv12_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 144)

# batch_norm12
batch_norm12_mean = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm1_running_mean.npy")
batch_norm12_variance = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm1_running_var.npy")
batch_norm12_scale = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm1_gamma.npy")
batch_norm12_bias = feed([1,144,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm1_beta.npy")
batch_norm12 = dml.batch_normalization(conv12, batch_norm12_mean, batch_norm12_variance, batch_norm12_scale, batch_norm12_bias, fused_activation=dml.FusedActivation.relu())

# conv13
conv13_filter = constant([32,144,1,1], "mobilenetv20_features_linearbottleneck3_conv2_weight.npy")
conv13_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv13 = dml.convolution(batch_norm12, conv13_filter, conv13_bias)

# batch_norm13
batch_norm13_mean = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm2_running_mean.npy")
batch_norm13_variance = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm2_running_var.npy")
batch_norm13_scale = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm2_gamma.npy")
batch_norm13_bias = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck3_batchnorm2_beta.npy")
batch_norm13 = dml.batch_normalization(conv13, batch_norm13_mean, batch_norm13_variance, batch_norm13_scale, batch_norm13_bias)

# conv14
conv14_filter = constant([192,32,1,1], "mobilenetv20_features_linearbottleneck4_conv0_weight.npy")
conv14_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv14 = dml.convolution(batch_norm13, conv14_filter, conv14_bias)

# batch_norm14
batch_norm14_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm0_running_mean.npy")
batch_norm14_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm0_running_var.npy")
batch_norm14_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm0_gamma.npy")
batch_norm14_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm0_beta.npy")
batch_norm14 = dml.batch_normalization(conv14, batch_norm14_mean, batch_norm14_variance, batch_norm14_scale, batch_norm14_bias, fused_activation=dml.FusedActivation.relu())

# conv15
conv15_filter = constant([192,1,3,3], "mobilenetv20_features_linearbottleneck4_conv1_weight.npy")
conv15_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv15 = dml.convolution(batch_norm14, conv15_filter, conv15_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm15
batch_norm15_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm1_running_mean.npy")
batch_norm15_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm1_running_var.npy")
batch_norm15_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm1_gamma.npy")
batch_norm15_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm1_beta.npy")
batch_norm15 = dml.batch_normalization(conv15, batch_norm15_mean, batch_norm15_variance, batch_norm15_scale, batch_norm15_bias, fused_activation=dml.FusedActivation.relu())

# conv16
conv16_filter = constant([32,192,1,1], "mobilenetv20_features_linearbottleneck4_conv2_weight.npy")
conv16_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv16 = dml.convolution(batch_norm15, conv16_filter, conv16_bias)

# batch_norm16
batch_norm16_mean = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm2_running_mean.npy")
batch_norm16_variance = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm2_running_var.npy")
batch_norm16_scale = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm2_gamma.npy")
batch_norm16_bias = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck4_batchnorm2_beta.npy")
batch_norm16 = dml.batch_normalization(conv16, batch_norm16_mean, batch_norm16_variance, batch_norm16_scale, batch_norm16_bias)

# add2
add2 = dml.add(batch_norm13, batch_norm16)

# conv17
conv17_filter = constant([192,32,1,1], "mobilenetv20_features_linearbottleneck5_conv0_weight.npy")
conv17_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv17 = dml.convolution(add2, conv17_filter, conv17_bias)

# batch_norm17
batch_norm17_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm0_running_mean.npy")
batch_norm17_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm0_running_var.npy")
batch_norm17_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm0_gamma.npy")
batch_norm17_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm0_beta.npy")
batch_norm17 = dml.batch_normalization(conv17, batch_norm17_mean, batch_norm17_variance, batch_norm17_scale, batch_norm17_bias, fused_activation=dml.FusedActivation.relu())

# conv18
conv18_filter = constant([192,1,3,3], "mobilenetv20_features_linearbottleneck5_conv1_weight.npy")
conv18_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv18 = dml.convolution(batch_norm17, conv18_filter, conv18_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm18
batch_norm18_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm1_running_mean.npy")
batch_norm18_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm1_running_var.npy")
batch_norm18_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm1_gamma.npy")
batch_norm18_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm1_beta.npy")
batch_norm18 = dml.batch_normalization(conv18, batch_norm18_mean, batch_norm18_variance, batch_norm18_scale, batch_norm18_bias, fused_activation=dml.FusedActivation.relu())

# conv19
conv19_filter = constant([32,192,1,1], "mobilenetv20_features_linearbottleneck5_conv2_weight.npy")
conv19_bias = graph.constant(np.zeros([1,32,1,1], np.float32))
conv19 = dml.convolution(batch_norm18, conv19_filter, conv19_bias)

# batch_norm19
batch_norm19_mean = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm2_running_mean.npy")
batch_norm19_variance = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm2_running_var.npy")
batch_norm19_scale = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm2_gamma.npy")
batch_norm19_bias = feed([1,32,1,1], "mobilenetv20_features_linearbottleneck5_batchnorm2_beta.npy")
batch_norm19 = dml.batch_normalization(conv19, batch_norm19_mean, batch_norm19_variance, batch_norm19_scale, batch_norm19_bias)

#add3
add3 = dml.add(add2, batch_norm19)

# conv20
conv20_filter = constant([192,32,1,1], "mobilenetv20_features_linearbottleneck6_conv0_weight.npy")
conv20_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv20 = dml.convolution(add3, conv20_filter, conv20_bias)

# batch_norm20
batch_norm20_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm0_running_mean.npy")
batch_norm20_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm0_running_var.npy")
batch_norm20_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm0_gamma.npy")
batch_norm20_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm0_beta.npy")
batch_norm20 = dml.batch_normalization(conv20, batch_norm20_mean, batch_norm20_variance, batch_norm20_scale, batch_norm20_bias, fused_activation=dml.FusedActivation.relu())

# conv21
conv21_filter = constant([192,1,3,3], "mobilenetv20_features_linearbottleneck6_conv1_weight.npy")
conv21_bias = graph.constant(np.zeros([1,192,1,1], np.float32))
conv21 = dml.convolution(batch_norm20, conv21_filter, conv21_bias, start_padding = [1,1], end_padding = [1,1], group_count = 192)

# batch_norm21
batch_norm21_mean = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm1_running_mean.npy")
batch_norm21_variance = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm1_running_var.npy")
batch_norm21_scale = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm1_gamma.npy")
batch_norm21_bias = feed([1,192,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm1_beta.npy")
batch_norm21 = dml.batch_normalization(conv21, batch_norm21_mean, batch_norm21_variance, batch_norm21_scale, batch_norm21_bias, fused_activation=dml.FusedActivation.relu())

# conv22
conv22_filter = constant([64,192,1,1], "mobilenetv20_features_linearbottleneck6_conv2_weight.npy")
conv22_bias = graph.constant(np.zeros([1,64,1,1], np.float32))
conv22 = dml.convolution(batch_norm21, conv22_filter, conv22_bias)

# batch_norm22
batch_norm22_mean = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm2_running_mean.npy")
batch_norm22_variance = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm2_running_var.npy")
batch_norm22_scale = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm2_gamma.npy")
batch_norm22_bias = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck6_batchnorm2_beta.npy")
batch_norm22 = dml.batch_normalization(conv22, batch_norm22_mean, batch_norm22_variance, batch_norm22_scale, batch_norm22_bias)

# conv23
conv23_filter = constant([384,64,1,1], "mobilenetv20_features_linearbottleneck7_conv0_weight.npy")
conv23_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv23 = dml.convolution(batch_norm22, conv23_filter, conv23_bias)

# batch_norm23
batch_norm23_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm0_running_mean.npy")
batch_norm23_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm0_running_var.npy")
batch_norm23_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm0_gamma.npy")
batch_norm23_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm0_beta.npy")
batch_norm23 = dml.batch_normalization(conv23, batch_norm23_mean, batch_norm23_variance, batch_norm23_scale, batch_norm23_bias, fused_activation=dml.FusedActivation.relu())

# conv24
conv24_filter = constant([384,1,3,3], "mobilenetv20_features_linearbottleneck7_conv1_weight.npy")
conv24_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv24 = dml.convolution(batch_norm23, conv24_filter, conv24_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm24
batch_norm24_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm1_running_mean.npy")
batch_norm24_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm1_running_var.npy")
batch_norm24_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm1_gamma.npy")
batch_norm24_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm1_beta.npy")
batch_norm24 = dml.batch_normalization(conv24, batch_norm24_mean, batch_norm24_variance, batch_norm24_scale, batch_norm24_bias, fused_activation=dml.FusedActivation.relu())

# conv25
conv25_filter = constant([64,384,1,1], "mobilenetv20_features_linearbottleneck7_conv2_weight.npy")
conv25_bias = graph.constant(np.zeros([1,64,1,1], np.float32))
conv25 = dml.convolution(batch_norm24, conv25_filter, conv25_bias)

# batch_norm25
batch_norm25_mean = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm2_running_mean.npy")
batch_norm25_variance = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm2_running_var.npy")
batch_norm25_scale = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm2_gamma.npy")
batch_norm25_bias = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck7_batchnorm2_beta.npy")
batch_norm25 = dml.batch_normalization(conv25, batch_norm25_mean, batch_norm25_variance, batch_norm25_scale, batch_norm25_bias)

# add4
add4 = dml.add(batch_norm22, batch_norm25)

# conv26
conv26_filter = constant([384,64,1,1], "mobilenetv20_features_linearbottleneck8_conv0_weight.npy")
conv26_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv26 = dml.convolution(add4, conv26_filter, conv26_bias)

# batch_norm26
batch_norm26_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm0_running_mean.npy")
batch_norm26_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm0_running_var.npy")
batch_norm26_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm0_gamma.npy")
batch_norm26_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm0_beta.npy")
batch_norm26 = dml.batch_normalization(conv26, batch_norm26_mean, batch_norm26_variance, batch_norm26_scale, batch_norm26_bias, fused_activation=dml.FusedActivation.relu())

# conv27
conv27_filter = constant([384,1,3,3], "mobilenetv20_features_linearbottleneck8_conv1_weight.npy")
conv27_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv27 = dml.convolution(batch_norm26, conv27_filter, conv27_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm27
batch_norm27_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm1_running_mean.npy")
batch_norm27_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm1_running_var.npy")
batch_norm27_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm1_gamma.npy")
batch_norm27_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm1_beta.npy")
batch_norm27 = dml.batch_normalization(conv27, batch_norm27_mean, batch_norm27_variance, batch_norm27_scale, batch_norm27_bias, fused_activation=dml.FusedActivation.relu())

# conv28
conv28_filter = constant([64,384,1,1], "mobilenetv20_features_linearbottleneck8_conv2_weight.npy")
conv28_bias = graph.constant(np.zeros([1,64,1,1], np.float32))
conv28 = dml.convolution(batch_norm27, conv28_filter, conv28_bias)

# batch_norm28
batch_norm28_mean = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm2_running_mean.npy")
batch_norm28_variance = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm2_running_var.npy")
batch_norm28_scale = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm2_gamma.npy")
batch_norm28_bias = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck8_batchnorm2_beta.npy")
batch_norm28 = dml.batch_normalization(conv28, batch_norm28_mean, batch_norm28_variance, batch_norm28_scale, batch_norm28_bias)

# add5
add5 = dml.add(add4, batch_norm28)

# conv29
conv29_filter = constant([384,64,1,1], "mobilenetv20_features_linearbottleneck9_conv0_weight.npy")
conv29_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv29 = dml.convolution(add5, conv29_filter, conv29_bias)

# batch_norm29
batch_norm29_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm0_running_mean.npy")
batch_norm29_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm0_running_var.npy")
batch_norm29_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm0_gamma.npy")
batch_norm29_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm0_beta.npy")
batch_norm29 = dml.batch_normalization(conv29, batch_norm29_mean, batch_norm29_variance, batch_norm29_scale, batch_norm29_bias, fused_activation=dml.FusedActivation.relu())

# conv30
conv30_filter = constant([384,1,3,3], "mobilenetv20_features_linearbottleneck9_conv1_weight.npy")
conv30_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv30 = dml.convolution(batch_norm29, conv30_filter, conv30_bias, start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm30
batch_norm30_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm1_running_mean.npy")
batch_norm30_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm1_running_var.npy")
batch_norm30_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm1_gamma.npy")
batch_norm30_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm1_beta.npy")
batch_norm30 = dml.batch_normalization(conv30, batch_norm30_mean, batch_norm30_variance, batch_norm30_scale, batch_norm30_bias, fused_activation=dml.FusedActivation.relu())

# conv31
conv31_filter = constant([64,384,1,1], "mobilenetv20_features_linearbottleneck9_conv2_weight.npy")
conv31_bias = graph.constant(np.zeros([1,64,1,1], np.float32))
conv31 = dml.convolution(batch_norm30, conv31_filter, conv31_bias)

# batch_norm31
batch_norm31_mean = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm2_running_mean.npy")
batch_norm31_variance = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm2_running_var.npy")
batch_norm31_scale = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm2_gamma.npy")
batch_norm31_bias = feed([1,64,1,1], "mobilenetv20_features_linearbottleneck9_batchnorm2_beta.npy")
batch_norm31 = dml.batch_normalization(conv31, batch_norm31_mean, batch_norm31_variance, batch_norm31_scale, batch_norm31_bias)

# add6
add6 = dml.add(add5, batch_norm31)

# conv32
conv32_filter = constant([384,64,1,1], "mobilenetv20_features_linearbottleneck10_conv0_weight.npy")
conv32_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv32 = dml.convolution(add6, conv32_filter, conv32_bias)

# batch_norm32
batch_norm32_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm0_running_mean.npy")
batch_norm32_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm0_running_var.npy")
batch_norm32_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm0_gamma.npy")
batch_norm32_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm0_beta.npy")
batch_norm32 = dml.batch_normalization(conv32, batch_norm32_mean, batch_norm32_variance, batch_norm32_scale, batch_norm32_bias, fused_activation=dml.FusedActivation.relu())

# conv33
conv33_filter = constant([384,1,3,3], "mobilenetv20_features_linearbottleneck10_conv1_weight.npy")
conv33_bias = graph.constant(np.zeros([1,384,1,1], np.float32))
conv33 = dml.convolution(batch_norm32, conv33_filter, conv33_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 384)

# batch_norm33
batch_norm33_mean = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm1_running_mean.npy")
batch_norm33_variance = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm1_running_var.npy")
batch_norm33_scale = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm1_gamma.npy")
batch_norm33_bias = feed([1,384,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm1_beta.npy")
batch_norm33 = dml.batch_normalization(conv33, batch_norm33_mean, batch_norm33_variance, batch_norm33_scale, batch_norm33_bias, fused_activation=dml.FusedActivation.relu())

# conv34
conv34_filter = constant([96,384,1,1], "mobilenetv20_features_linearbottleneck10_conv2_weight.npy")
conv34_bias = graph.constant(np.zeros([1,96,1,1], np.float32))
conv34 = dml.convolution(batch_norm33, conv34_filter, conv34_bias)

# batch_norm34
batch_norm34_mean = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm2_running_mean.npy")
batch_norm34_variance = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm2_running_var.npy")
batch_norm34_scale = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm2_gamma.npy")
batch_norm34_bias = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck10_batchnorm2_beta.npy")
batch_norm34 = dml.batch_normalization(conv34, batch_norm34_mean, batch_norm34_variance, batch_norm34_scale, batch_norm34_bias)

# conv35
conv35_filter = constant([576,96,1,1], "mobilenetv20_features_linearbottleneck11_conv0_weight.npy")
conv35_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv35 = dml.convolution(batch_norm34, conv35_filter, conv35_bias)

# batch_norm35
batch_norm35_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm0_running_mean.npy")
batch_norm35_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm0_running_var.npy")
batch_norm35_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm0_gamma.npy")
batch_norm35_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm0_beta.npy")
batch_norm35 = dml.batch_normalization(conv35, batch_norm35_mean, batch_norm35_variance, batch_norm35_scale, batch_norm35_bias, fused_activation=dml.FusedActivation.relu())

# conv36
conv36_filter = constant([576,1,3,3], "mobilenetv20_features_linearbottleneck11_conv1_weight.npy")
conv36_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv36 = dml.convolution(batch_norm35, conv36_filter, conv36_bias, start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm36
batch_norm36_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm1_running_mean.npy")
batch_norm36_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm1_running_var.npy")
batch_norm36_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm1_gamma.npy")
batch_norm36_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm1_beta.npy")
batch_norm36 = dml.batch_normalization(conv36, batch_norm36_mean, batch_norm36_variance, batch_norm36_scale, batch_norm36_bias, fused_activation=dml.FusedActivation.relu())

# conv37
conv37_filter = constant([96,576,1,1], "mobilenetv20_features_linearbottleneck11_conv2_weight.npy")
conv37_bias = graph.constant(np.zeros([1,96,1,1], np.float32))
conv37 = dml.convolution(batch_norm36, conv37_filter, conv37_bias)

# batch_norm37
batch_norm37_mean = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm2_running_mean.npy")
batch_norm37_variance = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm2_running_var.npy")
batch_norm37_scale = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm2_gamma.npy")
batch_norm37_bias = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck11_batchnorm2_beta.npy")
batch_norm37 = dml.batch_normalization(conv37, batch_norm37_mean, batch_norm37_variance, batch_norm37_scale, batch_norm37_bias)

# add7
add7 = dml.add(batch_norm34, batch_norm37)

# conv38
conv38_filter = constant([576,96,1,1], "mobilenetv20_features_linearbottleneck12_conv0_weight.npy")
conv38_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv38 = dml.convolution(add7, conv38_filter, conv38_bias)

# batch_norm38
batch_norm38_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm0_running_mean.npy")
batch_norm38_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm0_running_var.npy")
batch_norm38_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm0_gamma.npy")
batch_norm38_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm0_beta.npy")
batch_norm38 = dml.batch_normalization(conv38, batch_norm38_mean, batch_norm38_variance, batch_norm38_scale, batch_norm38_bias, fused_activation=dml.FusedActivation.relu())

# conv39
conv39_filter = constant([576,1,3,3], "mobilenetv20_features_linearbottleneck12_conv1_weight.npy")
conv39_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv39 = dml.convolution(batch_norm38, conv39_filter, conv39_bias, start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm39
batch_norm39_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm1_running_mean.npy")
batch_norm39_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm1_running_var.npy")
batch_norm39_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm1_gamma.npy")
batch_norm39_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm1_beta.npy")
batch_norm39 = dml.batch_normalization(conv39, batch_norm39_mean, batch_norm39_variance, batch_norm39_scale, batch_norm39_bias, fused_activation=dml.FusedActivation.relu())

# conv40
conv40_filter = constant([96,576,1,1], "mobilenetv20_features_linearbottleneck12_conv2_weight.npy")
conv40_bias = graph.constant(np.zeros([1,96,1,1], np.float32))
conv40 = dml.convolution(batch_norm39, conv40_filter, conv40_bias)

# batch_norm40
batch_norm40_mean = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm2_running_mean.npy")
batch_norm40_variance = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm2_running_var.npy")
batch_norm40_scale = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm2_gamma.npy")
batch_norm40_bias = feed([1,96,1,1], "mobilenetv20_features_linearbottleneck12_batchnorm2_beta.npy")
batch_norm40 = dml.batch_normalization(conv40, batch_norm40_mean, batch_norm40_variance, batch_norm40_scale, batch_norm40_bias)

# add8
add8 = dml.add(add7, batch_norm40)

# conv41
conv41_filter = constant([576,96,1,1], "mobilenetv20_features_linearbottleneck13_conv0_weight.npy")
conv41_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv41 = dml.convolution(add8, conv41_filter, conv41_bias)

# batch_norm41
batch_norm41_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm0_running_mean.npy")
batch_norm41_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm0_running_var.npy")
batch_norm41_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm0_gamma.npy")
batch_norm41_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm0_beta.npy")
batch_norm41 = dml.batch_normalization(conv41, batch_norm41_mean, batch_norm41_variance, batch_norm41_scale, batch_norm41_bias, fused_activation=dml.FusedActivation.relu())

# conv42
conv42_filter = constant([576,1,3,3], "mobilenetv20_features_linearbottleneck13_conv1_weight.npy")
conv42_bias = graph.constant(np.zeros([1,576,1,1], np.float32))
conv42 = dml.convolution(batch_norm41, conv42_filter, conv42_bias, strides = [2,2], start_padding = [1,1], end_padding = [1,1], group_count = 576)

# batch_norm42
batch_norm42_mean = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm1_running_mean.npy")
batch_norm42_variance = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm1_running_var.npy")
batch_norm42_scale = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm1_gamma.npy")
batch_norm42_bias = feed([1,576,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm1_beta.npy")
batch_norm42 = dml.batch_normalization(conv42, batch_norm42_mean, batch_norm42_variance, batch_norm42_scale, batch_norm42_bias, fused_activation=dml.FusedActivation.relu())

# conv43
conv43_filter = constant([160,576,1,1], "mobilenetv20_features_linearbottleneck13_conv2_weight.npy")
conv43_bias = graph.constant(np.zeros([1,160,1,1], np.float32))
conv43 = dml.convolution(batch_norm42, conv43_filter, conv43_bias)

# batch_norm43
batch_norm43_mean = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm2_running_mean.npy")
batch_norm43_variance = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm2_running_var.npy")
batch_norm43_scale = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm2_gamma.npy")
batch_norm43_bias = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck13_batchnorm2_beta.npy")
batch_norm43 = dml.batch_normalization(conv43, batch_norm43_mean, batch_norm43_variance, batch_norm43_scale, batch_norm43_bias)

# conv44
conv44_filter = constant([960,160,1,1], "mobilenetv20_features_linearbottleneck14_conv0_weight.npy")
conv44_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv44 = dml.convolution(batch_norm43, conv44_filter, conv44_bias)

# batch_norm44
batch_norm44_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm0_running_mean.npy")
batch_norm44_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm0_running_var.npy")
batch_norm44_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm0_gamma.npy")
batch_norm44_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm0_beta.npy")
batch_norm44 = dml.batch_normalization(conv44, batch_norm44_mean, batch_norm44_variance, batch_norm44_scale, batch_norm44_bias, fused_activation=dml.FusedActivation.relu())

# conv45
conv45_filter = constant([960,1,3,3], "mobilenetv20_features_linearbottleneck14_conv1_weight.npy")
conv45_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv45 = dml.convolution(batch_norm44, conv45_filter, conv45_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm45
batch_norm45_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm1_running_mean.npy")
batch_norm45_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm1_running_var.npy")
batch_norm45_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm1_gamma.npy")
batch_norm45_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm1_beta.npy")
batch_norm45 = dml.batch_normalization(conv45, batch_norm45_mean, batch_norm45_variance, batch_norm45_scale, batch_norm45_bias, fused_activation=dml.FusedActivation.relu())

# conv46
conv46_filter = constant([160,960,1,1], "mobilenetv20_features_linearbottleneck14_conv2_weight.npy")
conv46_bias = graph.constant(np.zeros([1,160,1,1], np.float32))
conv46 = dml.convolution(batch_norm45, conv46_filter, conv46_bias)

# batch_norm46
batch_norm46_mean = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm2_running_mean.npy")
batch_norm46_variance = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm2_running_var.npy")
batch_norm46_scale = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm2_gamma.npy")
batch_norm46_bias = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck14_batchnorm2_beta.npy")
batch_norm46 = dml.batch_normalization(conv46, batch_norm46_mean, batch_norm46_variance, batch_norm46_scale, batch_norm46_bias)

# add9
add9 = dml.add(batch_norm43, batch_norm46)

# conv47
conv47_filter = constant([960,160,1,1], "mobilenetv20_features_linearbottleneck15_conv0_weight.npy")
conv47_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv47 = dml.convolution(add9, conv47_filter, conv47_bias)

# batch_norm47
batch_norm47_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm0_running_mean.npy")
batch_norm47_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm0_running_var.npy")
batch_norm47_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm0_gamma.npy")
batch_norm47_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm0_beta.npy")
batch_norm47 = dml.batch_normalization(conv47, batch_norm47_mean, batch_norm47_variance, batch_norm47_scale, batch_norm47_bias, fused_activation=dml.FusedActivation.relu())

# conv48
conv48_filter = constant([960,1,3,3], "mobilenetv20_features_linearbottleneck15_conv1_weight.npy")
conv48_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv48 = dml.convolution(batch_norm47, conv48_filter, conv48_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm48
batch_norm48_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm1_running_mean.npy")
batch_norm48_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm1_running_var.npy")
batch_norm48_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm1_gamma.npy")
batch_norm48_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm1_beta.npy")
batch_norm48 = dml.batch_normalization(conv48, batch_norm48_mean, batch_norm48_variance, batch_norm48_scale, batch_norm48_bias, fused_activation=dml.FusedActivation.relu())

# conv49
conv49_filter = constant([160,960,1,1], "mobilenetv20_features_linearbottleneck15_conv2_weight.npy")
conv49_bias = graph.constant(np.zeros([1,160,1,1], np.float32))
conv49 = dml.convolution(batch_norm48, conv49_filter, conv49_bias)

# batch_norm49
batch_norm49_mean = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm2_running_mean.npy")
batch_norm49_variance = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm2_running_var.npy")
batch_norm49_scale = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm2_gamma.npy")
batch_norm49_bias = feed([1,160,1,1], "mobilenetv20_features_linearbottleneck15_batchnorm2_beta.npy")
batch_norm49 = dml.batch_normalization(conv49, batch_norm49_mean, batch_norm49_variance, batch_norm49_scale, batch_norm49_bias)

# add10
add10 = dml.add(add9, batch_norm49)

# conv50
conv50_filter = constant([960,160,1,1], "mobilenetv20_features_linearbottleneck16_conv0_weight.npy")
conv50_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv50 = dml.convolution(add10, conv50_filter, conv50_bias)

# batch_norm50
batch_norm50_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm0_running_mean.npy")
batch_norm50_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm0_running_var.npy")
batch_norm50_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm0_gamma.npy")
batch_norm50_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm0_beta.npy")
batch_norm50 = dml.batch_normalization(conv50, batch_norm50_mean, batch_norm50_variance, batch_norm50_scale, batch_norm50_bias, fused_activation=dml.FusedActivation.relu())

# conv51
conv51_filter = constant([960,1,3,3], "mobilenetv20_features_linearbottleneck16_conv1_weight.npy")
conv51_bias = graph.constant(np.zeros([1,960,1,1], np.float32))
conv51 = dml.convolution(batch_norm50, conv51_filter, conv51_bias, start_padding = [1,1], end_padding = [1,1], group_count = 960)

# batch_norm51
batch_norm51_mean = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm1_running_mean.npy")
batch_norm51_variance = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm1_running_var.npy")
batch_norm51_scale = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm1_gamma.npy")
batch_norm51_bias = feed([1,960,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm1_beta.npy")
batch_norm51 = dml.batch_normalization(conv51, batch_norm51_mean, batch_norm51_variance, batch_norm51_scale, batch_norm51_bias, fused_activation=dml.FusedActivation.relu())

# conv52
conv52_filter = constant([320,960,1,1], "mobilenetv20_features_linearbottleneck16_conv2_weight.npy")
conv52_bias = graph.constant(np.zeros([1,320,1,1], np.float32))
conv52 = dml.convolution(batch_norm51, conv52_filter, conv52_bias)

# batch_norm52
batch_norm52_mean = feed([1,320,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm2_running_mean.npy")
batch_norm52_variance = feed([1,320,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm2_running_var.npy")
batch_norm52_scale = feed([1,320,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm2_gamma.npy")
batch_norm52_bias = feed([1,320,1,1], "mobilenetv20_features_linearbottleneck16_batchnorm2_beta.npy")
batch_norm52 = dml.batch_normalization(conv52, batch_norm52_mean, batch_norm52_variance, batch_norm52_scale, batch_norm52_bias)

# conv53
conv53_filter = constant([1280,320,1,1], "mobilenetv20_features_conv1_weight.npy")
conv53_bias = graph.constant(np.zeros([1,1280,1,1], np.float32))
conv53 = dml.convolution(batch_norm52, conv53_filter, conv53_bias)

# batch_norm52
batch_norm53_mean = feed([1,1280,1,1], "mobilenetv20_features_batchnorm1_running_mean.npy")
batch_norm53_variance = feed([1,1280,1,1], "mobilenetv20_features_batchnorm1_running_var.npy")
batch_norm53_scale = feed([1,1280,1,1], "mobilenetv20_features_batchnorm1_gamma.npy")
batch_norm53_bias = feed([1,1280,1,1], "mobilenetv20_features_batchnorm1_beta.npy")
batch_norm53 = dml.batch_normalization(conv53, batch_norm53_mean, batch_norm53_variance, batch_norm53_scale, batch_norm53_bias, fused_activation=dml.FusedActivation.relu())

# avg_pool1
avg_pool1 = dml.average_pooling(batch_norm53, strides=[1,1], window_sizes=[7,7], start_padding=[0,0], end_padding=[0,0])

# conv54
conv54_filter = constant([1000,1280,1,1], "mobilenetv20_output_pred_weight.npy")
conv54 = dml.convolution(avg_pool1, conv54_filter)

# reshape
reshape = dml.reinterpret(conv54, [1,1,1,1000], [1000,1000,1000,1])

# softmax
soft_max = dml.activation_softmax(reshape)

op = graph.compile([soft_max])

output_tensor, = op({input: processed_image, **feeds})

# Opens text file of categories to collect the correct image category
label_file = open("imagenet_categories.txt","r")
label_lines = label_file.readlines()
prediction_index = np.argmax(output_tensor)

# Print the program confidence and the category from locally stored ImageNet text file
print("\nCategory: {}".format(label_lines[prediction_index], end=''))
print("Confidence: {:2.2f}%".format(np.amax(output_tensor) * 100))
