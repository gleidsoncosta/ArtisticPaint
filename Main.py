import cv2
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import caffe
import copy

kernel_size = 3
num_output = 3
stride = 1



def createNet():
    caffe.set_mode_cpu()
    net = caffe.Net('conv.prototxt', caffe.TEST)

    # create data - insert data to layer values
    im = cv2.imread('gatineos.jpg', 0)
    im_input = im[np.newaxis, np.newaxis, :, :]

    net.blobs['data'].reshape(*im_input.shape)
    net.blobs['data'].data[...] = im_input

    # initialize net
    net.forward()

    # get all layers on my net
    conv1 = net.blobs['conv1']
    conv2 = net.blobs['conv2']
    #pool1 = net.blobs['pool1']

    # process data according to to article

    # given an original and generated get respective feature representation
    input_feature_matrix = getAllFeatureMap(conv1)
    output_feature_matrix = getAllFeatureMap(conv2)

    #CONTENT
    # get error loss
    lcontent = getSquaredErrorLoss(input_feature_matrix, output_feature_matrix)
    # get the derivative
    matrix_derivative = getDerivativeLossMatrix(lcontent, input_feature_matrix, output_feature_matrix)

    #STYLE
    # calculus of gram matrix for input and output
    input_style = gramMatrix(input_feature_matrix)
    output_style = gramMatrix(output_feature_matrix)
    error = minMeanSqrDistanceContribution(input_style, output_style, output_feature_matrix.shape[1])
    #derivativeMinSqrDistanceContribution(error, output_feature_matrix, input_style, output_style)
    # minimising the mean-squared distance betweem gram matrixes



    b = reconstructFilterFromMap(0, matrix_derivative, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    c = reconstructFilterFromMap(0, input_feature_matrix, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    d = reconstructFilterFromMap(0, output_feature_matrix, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    cv2.imshow("derivative", b)
    cv2.imshow("input", c)
    cv2.imshow("output", d)
    cv2.imshow("direct", im)

    # saveImage("eita.jpg", b)
    # saveImage("kk.jpg", c)
    # saveImage("haha.jpg", d)

    net.save('mymodel.caffemodel')

    cv2.waitKey(0)

def mixImages():
    # To mix CONTENT OF PHOTOGRAPHY to STYLE OF PAINTING
    # Minimise the distance of a white noise image from CONTENT in ONE LAYER and STYLE in a NUMBER OF LAYERS
    # Ltotal(p,a,x) = alpha*Lcontent(p,x)+beta*Lstyle(a,x)
    # p     -> photography
    # a     -> artwork
    # alpha -> weighting factor
    # beta  -> weighting factor


    #
def derivativeMinSqrDistanceContribution(error, output_matrix, input_gram_matrix, output_gram_matrix):
    derivative = np.zeros((output_matrix.shape[0], output_matrix.shape[1]))
    print (output_matrix)
    new_output_matrix = copy.copy(output_matrix)
    new_output_matrix = new_output_matrix.transpose()

    for i in range(0, derivative.shape[0]):
        for j in range(0, derivative.shape[1]):
            if output_matrix[i][j] >= 0:
                derivative[i][j] = (float(1)/float((input_gram_matrix.shape[0]**2)*(output_matrix.shape[1]**2)))\
                                   *(float(new_output_matrix[j][i]*( output_matrix[j][i] - input_gram_matrix[j][i] )))
            else:
                derivative[i][j] = 0
    print(derivative)

def totalLossStyle(error):
    r_sum = 0
    w1 = 0
    for i in range(0, 3):
        r_sum = w1*error[i]

def minMeanSqrDistanceContribution(inp, out, sz_feat_map):
    r_sum = 0
    for i in range(0, inp.shape[0]):
        for j in range(0, inp.shape[1]):
            r_sum += (out[i][j] - inp[i][j])**2
    error = float(1)/float(((4*(inp.shape[0]**2)*(sz_feat_map**2)))*r_sum)
    return error

def gramMatrix(matrix):
    g_matrix = np.zeros((matrix.shape[0], matrix.shape[0]))
    result = 0
    for i in range(0, g_matrix.shape[0]):
        for j in range(0, g_matrix.shape[1]):
            a = []
            b = []
            for k in range(0, matrix.shape[1]):
                a.append(matrix[i][k])
                b.append(matrix[j][k])
            g_matrix[i][j] = np.inner(a, b)
    return g_matrix

def reconstructFilterFromMap(filter_pos, matrix, num_filters, shape):
    m = copy.copy(matrix)
    a = np.split(m, num_filters)
    b = np.reshape(a[filter_pos], shape)

    return b


def getDerivativeLossMatrix(lcontent, input_matrix, output_matrix):
    derivative = np.zeros((output_matrix.shape[0], output_matrix.shape[1]))
    for i in range(0, derivative.shape[0]):
        for j in range(0, derivative.shape[1]):
            if output_matrix[i][j] >= 0:
                derivative[i][j] = output_matrix[i][j] - input_matrix[i][j]
            else:
                derivative[i][j] = 0

    return derivative


def getSquaredErrorLoss(input_matrix, output_matrix):
    l_content_sum = 0
    for i in range(0, input_matrix.shape[0]):
        for j in range(0, input_matrix.shape[1]):
            l_content_sum += (output_matrix[i][j] - input_matrix[i][j]) ** 2
    l_content = 0.5 * l_content_sum

    return l_content


def getAllFeatureMap(conv):
    matrix = conv.data[0][0].flatten()
    for i in range(1, conv.shape[1]):
        matrix = np.vstack((matrix, conv.data[0][i].flatten()))
    return matrix


def showImage(text, image):
    cv2.imshow(text, image)


def saveImage(text, image):
    cv2.imwrite(text, 255 * image)


if __name__ == '__main__':
    createNet()
