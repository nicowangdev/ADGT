import numpy as np
import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=GradCAM(XXX)
        return

    def get_attribution_map(self, img, target=None):
        '''
                input:
                img: batch X channels X height X width [BCHW], torch floatTensor
                target:
                label:batch long Tensor

                output:
                attribution_map: batch X height X width,numpy
                '''
        if target is None:
            target = torch.argmax(self.model(img), 1)

        mask=self.explain(img)
        return mask