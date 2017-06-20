class Architectures():
    ''' Helper class that returns python dictionary containing
    shapes for CNN layers
    '''

    def xsmall(kernel_size, n_classes):
        return {'conv1' : [kernel_size, kernel_size, 1, 32],
                'conv2' : [kernel_size, kernel_size, 32, 64],
                'conv3' : [kernel_size, kernel_size, 64, 128],
                'conv4' : [kernel_size, kernel_size, 128, 128],
                'conv5' : [kernel_size, kernel_size, 128, 128],
                'conv5_1' : [kernel_size, kernel_size, 256, 512],
                'conv5_2' : [kernel_size, kernel_size, 512, 512],
                'fc1' : [0, 2048],
                'fc2' : [2048, n_classes]}

    def vgg16_skipped(kernel_size, n_classes):
        return {'conv1_1' : [kernel_size, kernel_size, 1, 64],
                'conv2_1' : [kernel_size, kernel_size, 64, 128],
                'conv3_1' : [kernel_size, kernel_size, 128, 256],
                'conv4_1' : [kernel_size, kernel_size, 256, 512],
                'conv5_1' : [kernel_size, kernel_size, 512, 512],
                'fc1' : [kernel_size*kernel_size*512, 4096],
                'fc2' : [4096, n_classes]}

    def vgg16(self):
        return {'conv1_1' : [kernel_size, kernel_size, 1, 64],
                'conv1_2' : [kernel_size, kernel_size, 64, 64],
        	    'conv2_1' : [kernel_size, kernel_size, 64, 128],
                'conv2_2' : [kernel_size, kernel_size, 128, 128],
                'conv3_1' : [kernel_size, kernel_size, 128, 256],
                'conv3_2' : [kernel_size, kernel_size, 256, 256],
                'conv3_3' : [kernel_size, kernel_size, 256, 256],
                'conv4_1' : [kernel_size, kernel_size, 256, 512],
                'conv4_2' : [kernel_size, kernel_size, 512, 512],
                'conv4_3' : [kernel_size, kernel_size, 512, 512],
                'conv5_1' : [kernel_size, kernel_size, 512, 512],
                'conv5_2' : [kernel_size, kernel_size, 512, 512],
                'conv5_3' : [kernel_size, kernel_size, 512, 512],
                'fc1' : [kernel_size*kernel_size*512, 4096],
                'fc2' : [4096, 4096],
                'fc3' : [4096, n_classes]}
