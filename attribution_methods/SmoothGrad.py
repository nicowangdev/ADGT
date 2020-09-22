from captum.attr import Saliency

import torch

class Explainer():
    def __init__(self,model,num_samples=50):
        self.model=model
        self.num_samples=num_samples
        self.explain=Saliency(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions=torch.zeros_like(img)
        max=torch.max(img)
        min=torch.min(img)
        for i in range(self.num_samples):
            attributions += self.explain.attribute(img+torch.randn_like(img)*0.1*(max-min), target=target,abs=False)
        return attributions
