from captum.attr import Saliency

import torch

class Explainer():
    def __init__(self,model,num_samples=10):
        self.model=model
        self.num_samples=num_samples
        self.explain=Saliency(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions=torch.zeros_like(img)
        for i in range(self.num_samples):
            attributions += self.explain.attribute(img+torch.randn_like(img)*0.001, target=target,abs=False)
        return attributions
