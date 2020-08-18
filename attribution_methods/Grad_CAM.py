from attribution_methods.gradcam_batch import *
import torch


class Explainer():
    def __init__(self, model, layer):
        self.model = model
        self.explainer = GradCam(model=model, feature_module=layer, use_cuda=torch.cuda.is_available())

    def get_attribution_map(self, img, target=None):
        attributions = self.explainer(input, target)
        return attributions


# test-passed
if __name__ == '__main__':
    """ python grad_cam_batch.py <path_to_image>
    1. Loads an image with opencv.
    2. Pre-processes it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    args = get_args()
    images = read_imgs(args.image_path)
    input = preprocess_imgs(images)

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    model = models.resnet50(pretrained=True)
    explainer = Explainer(model, model.layer4)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.

    target_index = None
    mask = explainer.get_attribution_map(input, target_index)
    show_cam_on_image(images, mask)


