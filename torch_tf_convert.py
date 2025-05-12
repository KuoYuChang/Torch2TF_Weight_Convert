import torch
import numpy as np

def convert(torch_list, tf_list, shape_list):
    '''
    convert weight from torch layers to tensorflow layers
    Without adding transpose layers

    Args:
        torch_list: list of torch layers, ordered as self.forword from torch model class
        tf_list:    list of tensorflow layers, ordered as self.call from tensorflow model class

                    * Note that layers without parameters, like pooling, no need to put into lists
        
        shape_list: list of shape of first linear/dense layer, only first layer encountered need, other linear layers no need
                    layer is first linear layer, put input shape into list, see from torchsummary
                    layers not first linear layer, or other types of layers, put None into list instead

                    eg : a torch model is like
                        ----------------------------------------------------------------
                                Layer (type)               Output Shape         Param #
                        ================================================================
                                    Conv2d-1            [-1, 6, 28, 28]             456
                                 MaxPool2d-2            [-1, 6, 14, 14]               0
                                    Conv2d-3           [-1, 16, 10, 10]           2,416
            get output shape---> MaxPool2d-4             [-1, 16, 5, 5]               0
            first linear ------>    Linear-5                  [-1, 120]          48,120
                                    Linear-6                   [-1, 84]          10,164
                                    Linear-7                   [-1, 10]             850
                        ================================================================

                        then we knows that Linear-5 is the first linear layer, and input shape is [16, 5, 5], 
                        which is output shape from previous layer

                        torch_list be like [net.conv1, net.conv2, net.linear1, net.linear2, net.linear3]
                        (follow member names in model class)

                        then shape_list be like: [None, None, (16, 5, 5), None, None]         


    '''
    
    num = len(torch_list)

    trans_permu = (0, 2, 3, 1)
    dense_permu = (1, 2, 0)

    for i in range(num):
        torch_layer = torch_list[i]
        tf_layer = tf_list[i]

        if type(torch_layer) == torch.nn.modules.conv.Conv2d:
            weight_np = torch_layer.weight.data.permute(2,3,1,0).cpu().detach().numpy()
            bias_np = torch_layer.bias.data.cpu().detach().numpy()

        elif type(torch_layer) == torch.nn.modules.linear.Linear:
            weight_np = torch_layer.weight.data.cpu().detach().numpy()
            bias_np = torch_layer.bias.data.cpu().detach().numpy()

            # for linear weight, regular transpose
            # tensorflow column major in Dense linear
            weight_np = np.transpose(weight_np)

            if shape_list[i] != None:
                shape_i = shape_list[i]
                dim = 1
                for size_j in shape_i:
                  dim = dim * size_j
                # prepare permutation index array
                permu = np.arange(dim)
                permu = np.reshape(permu, shape_list[i])
                permu_final = np.transpose(permu, axes=dense_permu)
                #permu_flat = tf.keras.backend.flatten(permu_final)
                permu_flat = permu_final.flatten()

                # column permutation on Dense weight matrix
                #tf.gather(arr, indices=permu_flat, axis=1 )
                weight_np = weight_np[permu_flat]

        tf_layer.set_weights([weight_np, bias_np])