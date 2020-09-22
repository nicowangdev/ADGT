from captum.attr import InputXGradient
import torch.nn as nn
import numpy as np
import torch,copy


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Function by subclassing
    torch.autograd.Function and implementing forward and backward passed
    which operate on tensors
    """
    def __init__(self,q=50):
        self.q=q

    def forward(self, input):
        """
        In the forward pass we receive a  Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input
        """

        input, = self.saved_tensors
        grad_input = grad_output.clone()
        a=input.clone()
        a[a<0]=0
        R=a*grad_output
        temp=R.view(a.size(0),-1).cpu().numpy()
        tau=np.percentile(temp,self.q,axis=1)
        for i in range(R.size(0)):
            grad_input[i,R[i] < tau[i]] = 0
        return grad_input

def convert_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child,nn.ReLU):
            setattr(model, child_name,MyReLU)
        else:
            convert_relu(child)

class Explainer():
    def __init__(self,model):
        self.model=copy.deepcopy(model)
        self.explain=InputXGradient(model)


    def get_attribution_map(self,img,target=None):
        if target is None:
            target=torch.argmax(self.model(img),1)
        attributions = self.explain.attribute(img, target=target)
        attributions[attributions<0]=0
        return attributions
