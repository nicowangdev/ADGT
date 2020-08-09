#from scipy.misc import imsave
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imsave
import torch.nn.functional as F
from torchvision import  transforms
plt.switch_backend('agg')
from utils import prepare_dataset
import numpy as np
import torch
import torch.nn as nn
from attribution_methods.gradcam import GradCam
from model import resnet
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
ROOT='./result/'
def get_preprocess_transform():
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([transforms.ToTensor()])
    return transf
preprocess_transform = get_preprocess_transform()
def save_images(X, save_path):
    # [0, 1] -> [0,255]

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
    X=(X-minn)/(maxx-minn)

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

    imsave(save_path, img)


def show_cam(raw_img,mask,filename):
    raw_img = raw_img.transpose(0, 2, 3, 1)
    raw_img = raw_img.reshape(raw_img.shape[1], raw_img.shape[2], raw_img.shape[3])
    plt.imshow(raw_img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.savefig(ROOT + dataset_name + filename)
    plt.close()

def show_dataset(train_loader,dataset_name,attack_state=0,image=None,target=None):
    if image is None:
        image = []
        target = []
        for data, label in train_loader:
            # print(data,label,index)
            for i in range(10):
                temp = data[label == i]
                temp2 = label[label == i]
                # temp2=torch.ones_like(label[0:10])*i
                image.append(temp[0:10])
                target.append(temp2[0:10])
            break
        image = torch.cat(tuple(image), 0)
        target = torch.cat(tuple(target), 0)
    if attack_state==1:
        if dataset_name == 'MNIST':
            attack = torch.zeros(10, 1, 28, 28)
            for i in range(10):
                attack[i, :, i * 2, 0] = 2
            image = F.relu(1 - F.relu(1 - (image + attack[target])))
        elif dataset_name == 'C10':
            attack = torch.zeros(10, 3, 32, 32)
            # mean,std=torch.Tensor(mean),torch.Tensor(std)
            # img=image*std+mean
            for i in range(10):
                attack[i, :, i * 2, 0] = 2
            image = F.relu(1 - F.relu(1 - (image + attack[target])))
    elif attack_state==2:
        if dataset_name == 'MNIST':
            print('equal to attack1')
            return
        attack = torch.zeros(10, 3, 32, 32)
        for i in range(10):
            attack[i, :, i * 2, 0] = 2
        for i in range(10):
            attack[i, :, i * 2, 0] = 2
            for j in range(10):
                attack[i, :, j * 2, 0] -= 1
        image = F.relu(1 - F.relu(1 - (image + attack[target])))
    save_images(image.numpy(), ROOT + dataset_name + str(attack_state)+'.jpg')
    return image,target

def vis_gradcam(dataset_name,attack_state,img,label):
    img=preprocess_transform(img)
    img=img.view(1,img.size(0),img.size(1),img.size(2))
    img.requires_grad_(True)
    print(img.size())
    model = torch.load(dataset_name + str(attack_state) + 'prenet.ckpt')
    model.eval()
    #print(model)

    grad_cam = GradCam(model=model, feature_module=model.conv4_x, target_layer_names=['1'], use_cuda=True)
    mask = grad_cam(img.cuda(), label)
    raw_img = img.data.numpy()
    show_cam(raw_img, mask, 'gradcam' + str(attack_state) + '.jpg')

def vis_lime(dataset_name,attack_state,img,label):
    raw_img = preprocess_transform(img)
    raw_img = raw_img.view(1, raw_img.size(0), raw_img.size(1), raw_img.size(2))
    raw_img = raw_img.numpy()
    model = torch.load(dataset_name + str(attack_state) + 'prenet.ckpt')

    def batch_predict(images):
        model.eval()

        batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device).float()

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    #test_pred = batch_predict([img])
    # =========================================================================================================
    #print(test_pred.squeeze().argmax())
    from lime import lime_image

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img.astype(np.float64),
                                             batch_predict, #labels=(label,),
                                             # classification function
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)  # number of images that will be sent to classification function

    from skimage.segmentation import mark_boundaries

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                hide_rest=True)
    #print(temp)
    print(mask)
    #temp=temp*mask.reshape(mask.shape[0],mask.shape[1],1)
    img_boundry1 = mark_boundaries(temp, mask)
    plt.imshow(img_boundry1)

    #show_cam(raw_img, mask, 'lime'+ str(attack_state) + '.jpg')
    plt.savefig(ROOT + dataset_name + 'lime'+ str(attack_state) + '.jpg')
    plt.close()

def vis_shap(dataset_name,attack_state,loader,img):
    from attribution_methods import shap
    raw_img=img.reshape([1,img.shape[0],img.shape[1],img.shape[2]])
    img2=img.reshape([1,img.shape[2],img.shape[0],img.shape[1]])
    img=raw_img
    for data, label in loader:
        x_train=data
    gpu_model = torch.load(dataset_name + str(attack_state) + 'prenet.ckpt')

    if dataset_name=='C10':
        model = resnet.resnet18(indim=3)
    elif dataset_name=='MNIST':
        model = resnet.resnet18(indim=1)
    model.load_state_dict(gpu_model.state_dict())
    model.eval()
    x_train=x_train.numpy()
    background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
    background=torch.Tensor(background)
    img2 = torch.Tensor(img2)
    e = shap.DeepExplainer(model, background)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values = e.shap_values(img2)
    shap_results=[]
    for kk in shap_values:
        shap_results.append(kk.reshape([kk.shape[0],kk.shape[2],kk.shape[3],kk.shape[1]]))
    # plot the feature attributions
    shap.image_plot(shap_results, raw_img)
    plt.savefig(ROOT + dataset_name + 'shap' + str(attack_state) + '.jpg')
    plt.close()

def vis_lrp(dataset_name,attack_state,img,label):
    from attribution_methods.innvestigator import InnvestigateModel
    img=preprocess_transform(img)
    img=img.view(1,img.size(0),img.size(1),img.size(2))
    print(img.size())
    gpu_model = torch.load(dataset_name + str(attack_state) + 'prenet.ckpt')
    #print(model)
    if dataset_name=='C10':
        model = resnet.resnet18(indim=3)
    elif dataset_name=='MNIST':
        model = resnet.resnet18(indim=1)
    model.load_state_dict(gpu_model.state_dict())
    model.eval()
    model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    inn_model = InnvestigateModel(model, lrp_exponent=1,
                                  method="b-rule",
                                  beta=0, epsilon=1e-6).cuda()

    def run_LRP(net, image_tensor):
        return inn_model.innvestigate(in_tensor=image_tensor, rel_for_class=1)

    AD_score, LRP_map = run_LRP(inn_model, img)
    #AD_score = AD_score[0][1].detach().cpu().numpy()
    LRP_map = LRP_map.detach().numpy().squeeze()
    mask = LRP_map
    print(mask.shape)
    raw_img = img.numpy()
    show_cam(raw_img, mask, 'lrp' + str(attack_state) + '.jpg')

def vis_integrated_gradient(dataset_name,attack_state,img,label):
    from attribution_methods.integrated_gradients import random_baseline_integrated_gradients
    from attribution_methods.util import calculate_outputs_and_gradients,generate_entrie_images
    temp=preprocess_transform(img)
    raw_img=temp.view(1,temp.size(0),temp.size(1),temp.size(2))
    img = img.astype(np.float32)
    img = img[:, :, (2, 1, 0)]
    print(img.shape)
    model = torch.load(dataset_name + str(attack_state) + 'prenet.ckpt')
    model.eval()
    from utils.visualization import visualize
    gradients, label_index = calculate_outputs_and_gradients([img], model, None, True)
    gradients = np.transpose(gradients[0], (1, 2, 0))
    img_gradient_overlay = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=True,
                                     mask_mode=True)
    img_gradient = visualize(gradients, img, clip_above_percentile=99, clip_below_percentile=0, overlay=False)
    # calculate the integrated gradients
    attributions = random_baseline_integrated_gradients(img, model, label_index, calculate_outputs_and_gradients,
                                                        steps=50, num_random_trials=10, cuda=True)

    img_integrated_gradient_overlay = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0,
                                                overlay=True, mask_mode=True)
    img_integrated_gradient = visualize(attributions, img, clip_above_percentile=99, clip_below_percentile=0,
                                        overlay=False)
    output_img = generate_entrie_images(img, img_gradient, img_gradient_overlay, img_integrated_gradient,
                                        img_integrated_gradient_overlay)
    cv2.imwrite(ROOT+ dataset_name+'integrated_gradient' + str(attack_state) + '.jpg', np.uint8(output_img))


def main_gradcam(dataset_name='C10'):
    index = 1
    train_set = prepare_dataset.get_train_dataset(name=dataset_name, aug=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)

    attack_state = 0
    image, target = show_dataset(train_loader, dataset_name, attack_state)
    img = image[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img=img.permute(0,2,3,1).squeeze()
    print('obtain attack:', attack_state)
    tt = target[index].item()
    print(tt)
    vis_gradcam(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 1
    image1,_ = show_dataset(train_loader, dataset_name, attack_state,image,target)
    img = image1[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_gradcam(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 2
    image2, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image2[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_gradcam(dataset_name, attack_state, img.numpy(), tt)
def main_lime(dataset_name='C10'):
    index = 1
    train_set = prepare_dataset.get_train_dataset(name=dataset_name, aug=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)

    attack_state = 0
    image, target = show_dataset(train_loader, dataset_name, attack_state)
    img = image[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    tt = target[index].item()
    print(tt)
    vis_lime(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 1
    image1, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image1[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_lime(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 2
    image2, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image2[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_lime(dataset_name, attack_state, img.numpy(), tt)
def main_shap(dataset_name='C10'):
    index = 1
    train_set = prepare_dataset.get_train_dataset(name=dataset_name, aug=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)

    attack_state = 0
    image, target = show_dataset(train_loader, dataset_name, attack_state)
    img = image[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    tt = target[index].item()
    print(tt)
    vis_shap(dataset_name, attack_state,train_loader, img.numpy())
    # ======================================================================================
    attack_state = 1
    image1, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image1[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_shap(dataset_name, attack_state,train_loader, img.numpy())
    # ======================================================================================
    attack_state = 2
    image2, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image2[index].view(1, image.size(1), image.size(2), image.size(3))  # .requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_shap(dataset_name, attack_state,train_loader, img.numpy())
def main_lrp(dataset_name='C10'):
    index = 1
    train_set = prepare_dataset.get_train_dataset(name=dataset_name, aug=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)

    attack_state = 0
    image, target = show_dataset(train_loader, dataset_name, attack_state)
    img = image[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img=img.permute(0,2,3,1).squeeze()
    print('obtain attack:', attack_state)
    tt = target[index].item()
    print(tt)
    vis_lrp(dataset_name, attack_state, img.numpy(), tt)
def main_integrated_gradient(dataset_name='C10'):
    index = 1
    train_set = prepare_dataset.get_train_dataset(name=dataset_name, aug=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True, num_workers=4)

    attack_state = 0
    image, target = show_dataset(train_loader, dataset_name, attack_state)
    img = image[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img=img.permute(0,2,3,1).squeeze()
    print('obtain attack:', attack_state)
    tt = target[index].item()
    print(tt)
    vis_integrated_gradient(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 1
    image1,_ = show_dataset(train_loader, dataset_name, attack_state,image,target)
    img = image1[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_integrated_gradient(dataset_name, attack_state, img.numpy(), tt)
    # ======================================================================================
    attack_state = 2
    image2, _ = show_dataset(train_loader, dataset_name, attack_state, image, target)
    img = image2[index].view(1, image.size(1), image.size(2), image.size(3))#.requires_grad_(True)
    img = img.permute(0, 2, 3, 1).squeeze()
    print('obtain attack:', attack_state)
    vis_integrated_gradient(dataset_name, attack_state, img.numpy(), tt)

if __name__ == '__main__':
    dataset_name='C10'
    main_integrated_gradient(dataset_name)



