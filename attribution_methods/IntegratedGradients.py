from captum.attr import IntegratedGradients

import torch

class Explainer():
    def __init__(self,model):
        self.model=model
        self.explain=IntegratedGradients(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        baseline_dist = torch.randn_like(img) * 0.001
        thrd=20
        attributions=[]
        if img.size(0) > thrd:
            img=torch.split(img,thrd)
            target=torch.split(target,thrd)
            baseline_dist=torch.split(baseline_dist,thrd)
        else:
            img,target,baseline_dist=[img],[target],[baseline_dist]

        for i,t,b in zip(img,target,baseline_dist):
            temp = self.explain.attribute(i, b, target=t , return_convergence_delta=False)
            attributions.append(temp)

        attributions=torch.cat(tuple(attributions),0)
        return attributions