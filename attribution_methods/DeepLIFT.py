from captum.attr import DeepLift

import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=DeepLift(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        baseline_dist = torch.randn_like(img) * 0.001
        attributions, delta = self.explain.attribute(img, baseline_dist, target=target, return_convergence_delta=True)
        return attributions