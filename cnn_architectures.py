class Architectures():
    ''' Helper class that returns python dictionary containing
    shapes for CNN layers
    '''

    def vgg16_downsized(kernel_size, amount_classes):
        architecture = {'conv1_1' : [kernel_size, kernel_size, 1, 64],
                        'conv2_1' : [kernel_size, kernel_size, 64, 128],
                        'conv3_1' : [kernel_size, kernel_size, 128, 256],
                        'conv4_1' : [kernel_size, kernel_size, 256, 512],
                        'conv5_1' : [kernel_size, kernel_size, 512, 512],
                        'fc6' : [kernel_size*kernel_size*512, 4096],
                        'fc7' : [4096, amount_classes]}
        return architecture

    def vgg16(self):
        '''conv1_1 [3, 3, 3, 64]
        conv1_2 [3, 3, 64, 64]
        pool1
        conv2_1 [3, 3, 64, 128]
        conv2_2 [3, 3, 128, 128]
        pool2
        conv3_1 [3, 3, 128, 256]
        conv3_2 [3, 3, 256, 256]
        conv3_3 [3, 3, 256, 256]
        pool3
        conv4_1 [3, 3, 256, 512]
        conv4_2 [3, 3, 512, 512]
        conv4_3 [3, 3, 512, 512]
        pool4
        conv5_1 [3, 3, 512, 512]
        conv5_2 [3, 3, 512, 512]
        conv5_3 [3, 3, 512, 512]
        pool5
        fc1 [shape, 4096]
        fc2 [4096, 4096]
        fc3 [4096, classes]'''
        pass
