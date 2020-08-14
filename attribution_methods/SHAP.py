from attribution_methods import explainer
from captum.attr import (
    GradientShap,
    DeepLiftShap,
)
import collections
import numpy as np
import torch

class gradient_shap():
    def __init__(self,model,stdevs=0.09, n_samples=4):
        self.model=model
        self.explain=GradientShap(model)
        self.stdevs=stdevs
        self.n_samples=n_samples

    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        baseline_dist=torch.randn_like(img)*0.001
        attributions, delta = self.explain.attribute(img, stdevs=self.stdevs, n_samples=self.n_samples, baselines=baseline_dist,
                                   target=target, return_convergence_delta=True)
        return attributions

class deeplift_shap():
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
        attributions, delta = self.explain.attribute(input, baseline_dist, target=target, return_convergence_delta=True)
        return attributions