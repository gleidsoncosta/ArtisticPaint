import cv2
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import caffe
import copy
from scipy.optimize import minimize

kernel_size = 3
num_output = 10
stride = 1

WEIGHTS = { "content": {"conv1": 1.0},
            "style": {  "conv2": 0.2,
                        "conv3": 0.2}}

def createNet():
    caffe.set_mode_cpu()
    net = caffe.Net('conv.prototxt', caffe.TEST)

    # create data - insert data to layer values
    content = cv2.imread('gatineos.jpg', 0)
    content_input = content[np.newaxis, np.newaxis, :, :]

    style = cv2.imread('stnight.jpeg', 0)
    style_input = style[np.newaxis, np.newaxis, :, :]

    net.blobs['data'].reshape(*content_input.shape)
    net.blobs['data'].data[...] = content_input

    weights = WEIGHTS

    layers = []
    for layer in net.blobs:
        layers.append(layer)

    merge = (1*content)+(1 * style)
    g_style = getRepresentationStyle(style, net, weights["style"].keys())
    f_content = getRepresentationContent(content, net, weights["content"].keys())

    # optimization params
    grad_method = "L-BFGS-B"
    reprs = (g_style, f_content)
    minfn_args = {
        "args": (net, weights, layers, reprs, 1e5),
        "method": grad_method, "jac": True,
        "options": {"maxcor": 8, "maxiter": 100}
    }
    #minimization(merge.flatten(), net, weights, layers, reprs, 1e5)
    res = minimize(minimization, merge.flatten(), **minfn_args).nit
    cv2.imshow("Content", content)
    cv2.imshow("Style", style)
    cv2.imshow("Merge", merge)
    cv2.imshow("Resultado", net.blobs["data"].data[0][0])


    # cv2.imshow("B", net.blobs["data"].data[0])
    # # initialize net
    # net.forward()
    #
    # # get all layers on my net
    # conv1 = net.blobs['conv1']
    # conv2 = net.blobs['conv2']
    # #pool1 = net.blobs['pool1']
    #
    # # process data according to to article
    #
    # # given an original and generated get respective feature representation
    # input_feature_matrix = getAllFeatureMap(conv1)
    # output_feature_matrix = getAllFeatureMap(conv2)
    #
    #
    # #CONTENT
    # # get error loss
    # lcontent = getSquaredErrorLoss(input_feature_matrix, output_feature_matrix)
    # # get the derivative
    # content_derivative = getDerivativeLossMatrix(lcontent, input_feature_matrix, output_feature_matrix)
    #
    # #STYLE
    # # calculus of gram matrix for input and output
    # input_style = gramMatrix(input_feature_matrix)
    # output_style = gramMatrix(output_feature_matrix)
    # error = minMeanSqrDistanceContribution(input_style, output_style, output_feature_matrix.shape[1])
    # style_derivative = derivativeMinSqrDistanceContribution(error, output_feature_matrix, input_style, output_style)
    # # minimising the mean-squared distance betweem gram matrixes
    #
    # cv2.imshow("A", conv1.data[0][0])
    # conv1.diff[0] += reconstructFilterFromMap(0, content_derivative, num_output,
    #                                           (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    # net.backward(start=1, end=0)
    # cv2.imshow("B", conv1.data[0][0])
    #
    #
    # a = reconstructFilterFromMap(0, content_derivative, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    # b = reconstructFilterFromMap(0, style_derivative, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    # c = reconstructFilterFromMap(0, input_feature_matrix, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    # d = reconstructFilterFromMap(0, output_feature_matrix, num_output, (conv1.data[0][0].shape[0], conv1.data[0][0].shape[1]))
    # cv2.imshow("content_derivative", a)
    # cv2.imshow("style_derivative", b)
    # cv2.imshow("input", c)
    # cv2.imshow("output", d)
    # cv2.imshow("original", content)
    #
    # # saveImage("eita.jpg", b)
    # # saveImage("kk.jpg", c)
    # # saveImage("haha.jpg", d)
    #
    # net.save('mymodel.caffemodel')

    cv2.waitKey(0)

def minimization(flatten, net, weights, layers, reprs, ratio):
    layers_style = weights["style"].keys()
    layers_content = weights["content"].keys()

    img = copy.copy(flatten)
    init = img.reshape( (360,480) )
    #g_style -> List of matrices with all filters in one layer
    #f_content -> all filters in one layer

    #initial representation
    (g_style_input, f_content_input) = reprs
    #current representation
    f_content_output = getRepresentationContent(init, net, weights["content"])
    g_style_output = getRepresentationStyle(init, net, weights["style"])

    loss = 0
    net.blobs[layers[-1]].diff[:] = 0
    for i, layer in enumerate(reversed(layers)):
        next_layer = None
        if i >= len(layers)-1:
            next_layer = None
        else:
            next_layer = layers[-i - 2]

        print(layer, i, next_layer)

        grad = net.blobs[layer].diff[0]

        if layer in layers_style:
            wl = weights["style"][layer]
            offset = len(layers)-len(g_style_output)
            out_gram_matrix = gramMatrix(g_style_output[offset-1-i])
            inp_gram_matrix = gramMatrix(g_style_input[offset-1-i])
            l = minMeanSqrDistanceContribution(inp_gram_matrix, out_gram_matrix, g_style_output[offset-1-i].shape[1]) * wl * ratio
            g = derivativeMinSqrDistanceContribution(l, g_style_output[offset-1-i], inp_gram_matrix, out_gram_matrix)
            loss += wl*l
            grad += wl*g.reshape(grad.shape)*ratio

        # content contribution
        if layer in layers_content:
            wl = weights["content"][layer]
            l = getSquaredErrorLoss(f_content_input, f_content_output)
            g = getDerivativeLossMatrix(l, f_content_input, f_content_output)
            loss += wl*l
            grad += wl*g.reshape(grad.shape)

        # compute gradient
        #net.backward(start=layer, end=next_layer)
        if next_layer is None:
            grad = net.blobs["data"].diff[0]
        else:
            grad = net.blobs[next_layer].diff[0]
    net.backward()

    # format gradient for minimize() function
    grad = grad.flatten().astype(np.float64)
    return loss, grad

def getRepresentationContent(img_init, net, layer):
    net.blobs["data"].data[0] = img_init
    net.forward()
    for l in layer:
        output_matrix = getAllFeatureMap(net.blobs[l])
    return output_matrix

def getRepresentationStyle(img_init, net, layers):
    net.blobs["data"].data[0] = img_init
    net.forward()
    all = []
    for l in layers:
        all.append(getAllFeatureMap(net.blobs[l]))
    return all


def derivativeMinSqrDistanceContribution(error, feature_matrix, input_gram_matrix, output_gram_matrix):
    fmt = copy.copy(feature_matrix)
    derivative = np.zeros((fmt.shape[0], fmt.shape[1]))
    feat_mat_transpose = fmt.transpose()

    mat_result = np.zeros((feat_mat_transpose.shape[0], output_gram_matrix.shape[1]))
    for i in range(0, mat_result.shape[0]):
        for j in range(0, mat_result.shape[1]):
            for k in range(0, mat_result.shape[1]):
                #print(i, j, k)
                mat_result[i][j] += feat_mat_transpose[i][k]*(output_gram_matrix[k][j]-input_gram_matrix[k][j])

    # result from mat_result return as feature_matrix transpose size

    for i in range(0, derivative.shape[0]):
        for j in range(0, derivative.shape[1]):
            if fmt[i][j] >= 0:
                derivative[i][j] = float((float(1)/float((output_gram_matrix.shape[0]**2)*(feature_matrix.shape[1]**2)))*mat_result[j][i])

    return derivative

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
