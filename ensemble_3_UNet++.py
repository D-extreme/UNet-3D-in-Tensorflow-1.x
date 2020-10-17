"""
Created on Tuesday Oct 14 2020

@author: Shashwat Pathak 
"""
import tensorflow as tf
import numpy as np
from conv import *

class Unetplusplus3D():
    def __init__():
        self.inputI = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,256,256,128, self.inputI_chn], name='inputI')

    def encoder_decoder(self, inputI):
        #Backbone
        print("Backbone")
        conv0_0 = conv_bn_relu(input = inputI, output_chn = 32, stride = (1,1,1),isTraining=phase_flag, use_bias = True, name= "conv0_0")
        max0_0 = tf.keras.layers.MaxPool3D(pool_size = (2,2,2))(conv0_0)
        print("conv0_0:",np.shape(max0_0))
        conv1_0 = conv_bn_relu(input = max0_0, output_chn = 64, stride = (1,1,1),isTraining=phase_flag, use_bias = True, name= "conv1_0")
        max1_0 = tf.keras.layers.MaxPool3D(pool_size = (2,2,2))(conv1_0)
        print("conv1_0:",np.shape(max1_0))
        conv2_0 = conv_bn_relu(input = max1_0, output_chn = 128, stride = (1,1,1),isTraining=phase_flag, use_bias = True, name= "conv2_0")
        max2_0 = tf.keras.layers.MaxPool3D(pool_size = (2,2,2))(conv2_0)
        print("conv2_0:",np.shape(max2_0))

        #Central
        print("Central")
        conv3_0 = conv_bn_relu(input = max2_0, output_chn = 256, stride = (1,1,1),isTraining=phase_flag, use_bias = True, name= "conv3_0")
        max3_0 = tf.keras.layers.MaxPool3D(pool_size = (2,2,2))(conv3_0)
        print("conv3_0:",np.shape(max0_0))

        #Encodings and Concatenations
        print("Decoder")
        deconv1_0 = deconv_bn_relu(input = conv1_0, output_chn = 128, name = "deconv1_0", isTraining = phase_flag)
        conv0_1 = tf.concat([conv0_0,deconv1_0])
        print("conv0_1:",np.shape(conv0_1))

        deconv2_0 = deconv_bn_relu(input = conv2_0, output_chn = 128, name = "deconv2_0", isTraining = phase_flag)
        conv1_1 = tf.concat([conv1_0,deconv2_0])
        print("conv1_1:",np.shape(conv1_1))

        deconv1_1 = deconv_bn_relu(input = conv1_1, output_chn = 64, name = "deconv1_1", isTraining = phase_flag)
        conv0_2 = tf.concat([conv0_1, deconv1_1, conv0_0])
        print("conv0_2:",np.shape(conv0_2))

        deconv3_0 = deconv_bn_relu(input = conv3_0, output_chn = 128, name = "deconv3_0", isTraining = phase_flag)
        conv2_1 = tf.concat([conv2_0, deconv3_0])
        print("conv2_1:",np.shape(conv2_1))

        deconv2_1 = deconv_bn_relu(input = conv2_1, output_chn = 64, name = "deconv2_1", isTraining = phase_flag)
        conv1_2 = tf.concat([conv1_1, deconv2_1, conv1_0])
        print("conv1_2:",np.shape(conv1_2))

        deconv1_2 = deconv_bn_relu(input = conv1_2, output_chn = 32, name = "deconv1_2", isTraining = phase_flag)
        conv0_3 = tf.concat([deconv1_2, conv0_0, conv0_1, conv0_3])
        print("conv0_3:",np.shape(conv0_3))

        conv_result = conv3d(input = conv0_3, output_chn =2, kernel_size = 1, name = "conv_result")
        print("conv_result:",np.shape(conv_result))

        soft_prob = tf.nn.softmax(, name="label")
        label = tf.nn.argmax(soft_prob)

        return label, soft_prob
