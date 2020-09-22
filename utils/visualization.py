import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import  transforms
plt.switch_backend('agg')
import torch
import torch.nn as nn

def show_cam(raw_img,mask,filename):
    raw_img = raw_img.transpose(0, 2, 3, 1)
    raw_img = raw_img.reshape(raw_img.shape[1], raw_img.shape[2], raw_img.shape[3])
    plt.imshow(raw_img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.savefig(filename)
    plt.close()

def save_images(X, save_path,minn=None,maxx=None):
    # [0, 1] -> [0,255]

    if minn is None:
        minn=np.min(X.reshape([X.shape[0],-1]),axis=1)
        maxx=np.max(X.reshape([X.shape[0],-1]),axis=1)
        if X.ndim == 4:
            minn=minn.reshape([X.shape[0],1,1,1])
            maxx=maxx.reshape([X.shape[0],1,1,1])
        elif X.ndim==3:
            minn = minn.reshape([X.shape[0], 1, 1])
            maxx = maxx.reshape([X.shape[0], 1, 1])
        else :
            minn = minn.reshape([X.shape[0], 1])
            maxx = maxx.reshape([X.shape[0], 1])

        X=(X-minn)/(maxx-minn+1e-8)
    else:
        X = (X - minn) / (maxx - minn + 1e-8)
        X=np.maximum(X,0)
        X=np.minimum(X,1)
    #X = X.squeeze()
    #if isinstance(X.flatten()[0], np.floating):
    #    #X = (255 * X).astype('uint8')
    #    X=np.uint8(255 * X)

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = int(rows), int(n_samples / rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w] = x
    np.set_printoptions(threshold=np.inf)
    #print(save_path,img)
    plt.imsave(save_path, img)
    plt.close()


import numpy as np
import cv2

G = [0, 255, 0]
R = [255, 0, 0]


def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2,
                     plot_distribution=False):
    m = compute_threshold_by_top_percentage(attributions, percentage=100 - clip_above_percentile,
                                            plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(attributions, percentage=100 - clip_below_percentile,
                                            plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed


def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    if plot_distribution:
        raise NotImplementedError
    return threshold


def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)


def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, morphological_cleanup=False,
              structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=90, overlay=True,
              mask_mode=False, plot_distribution=False):
    if polarity == 'both':
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile, clip_below_percentile, 0.0,
                                    plot_distribution=plot_distribution)
    attributions_mask = attributions.copy()
    if morphological_cleanup:
        raise NotImplementedError
    if outlines:
        raise NotImplementedError
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 255)
            attributions = attributions[:, :, (2, 1, 0)]
    return attributions