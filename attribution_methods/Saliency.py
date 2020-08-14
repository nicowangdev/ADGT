from captum.attr import Saliency

import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=Saliency(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions = self.explain.attribute(img, target=target)
        return attributions
