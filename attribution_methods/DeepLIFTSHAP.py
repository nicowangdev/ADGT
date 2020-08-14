from captum.attr import DeepLiftShap
import collections
import numpy as np
import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=DeepLiftShap(model)
        return
    def get_attribution_map(self,img,target=None):
        '''
        input:
        img: batch X channels X height X width [BCHW], torch Tensor

        output:
        attribution_map: batch X height X width,numpy
        '''
        baseline_dist = torch.randn_like(img) * 0.001
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions, delta = self.explain.attribute(img, baseline_dist, target=target, return_convergence_delta=True)
        return attributions