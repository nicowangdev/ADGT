import numpy as np
import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=GradCAM(XXX)
        return
    def get_attribution_map(self,img):
        '''
        input:
        img: batch X channels X height X width [BCHW], torch Tensor

        output:
        attribution_map: batch X height X width,numpy
        '''
        mask=self.explain(img)
        return mask