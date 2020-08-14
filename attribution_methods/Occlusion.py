from captum.attr import Occlusion

import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=Occlusion(model)


    def get_attribution_map(self,img,target=None,sliding_window_shapes=(3,3,3)):
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions = self.explain.attribute(img, target=target,sliding_window_shapes=sliding_window_shapes)
        return attributions
