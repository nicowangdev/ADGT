import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers if target_layers else []
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        self.gradients = []
        items = self.model._modules.items()
        i = 1
        for name, module in items:
            x = module(x)
            if (i == len(items)) or (name in self.target_layers):
                x.register_hook(self.save_gradient)
                outputs = [x]
            i = i + 1
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" == name.lower() or 'avg_pool' == name.lower():
                x = module(x)
                x = x.view(x.size(0), -1)
            else:
                x = module(x)

        return target_activations, x


class GradCam:
    def __init__(self, model, feature_module, target_layer_names=None, use_cuda=None):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        # use cuda?
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        # get the class with the highest probity
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # get the one-hot code of the output
        one_hot = np.zeros((output.size()[0], output.size()[-1]), dtype=np.float32)
        for i in range(one_hot.shape[0]):
            one_hot[i][index[i]] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # clear grad and backward
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # get grads
        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        # get the activation of the target layer
        target = features[-1]
        target = target.cpu().data.numpy()
        # print("target shape", target.shape)
        weights = np.mean(grads_val, axis=(2, 3))

        # compute the cam
        cam = np.zeros((target.shape[0], target.shape[2], target.shape[3]), dtype=np.float32)
        for i in range(cam.shape[0]):
            for j, w in enumerate(weights[i]):
                cam[i] += w * target[i, j, :, :]
        cam = np.maximum(cam, 0)

        # 上采样
        new_cam = np.zeros((input.shape[0], input.shape[2], input.shape[3]))
        for i in range(new_cam.shape[0]):
            new_cam[i] = cv2.resize(cam[i], (input.shape[2], input.shape[3]))
        cam = new_cam
        cam = cam / (np.max(cam) + 1e-8)
        return torch.Tensor(cam.reshape([cam.shape[0],1,cam.shape[1],cam.shape[2]]))


# test-passed
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='../examples',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


# test-passed
def read_imgs(path):
    """
    read images from given folder
    :param path: the folder path
    :return: images, BHWC
    """
    dirs = os.listdir(path)
    imgs = []
    for fn in dirs:
        img_path = path + '/' + fn
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


# test-passed
def preprocess_imgs(imgs):
    """
    pre-process the images
    :param imgs: images, BHWC
    :return: input (processed images) , BCHW
    """
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_imgs = imgs.copy()[:, :, :, ::-1]
    for i in range(3):
        preprocessed_imgs[:, :, :, i] = preprocessed_imgs[:, :, :, i] - means[i]
        preprocessed_imgs[:, :, :, i] = preprocessed_imgs[:, :, :, i] / stds[i]
    preprocessed_imgs = \
        np.ascontiguousarray(np.transpose(preprocessed_imgs, (0, 3, 1, 2)))
    preprocessed_imgs = torch.from_numpy(preprocessed_imgs)
    input = preprocessed_imgs.requires_grad_(True)
    return input


# test-passed
def show_cam_on_image(imgs, masks, file_name="cam1.jpg"):
    """
    展示最终结果，这里只做了单个图片的展示
    :param imgs: imgs ,BHWC
    :param masks: attributions , BWC
    :param file_name: 最后结果保存的文件
    """
    plt.imshow(imgs[3])
    plt.imshow(masks[3], cmap="jet", alpha=0.5)
    plt.savefig(file_name)
