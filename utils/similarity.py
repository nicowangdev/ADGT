import numpy as np
import cv2
import os


def get_cov(x, y):
    n = len(x)
    x_bar = np.mean(x)
    y_bar = np.mean(y)
    var = np.sum((x - x_bar) * (y - y_bar)) / (n - 1)
    return var


class similarity:
    def __init__(self, figs1, figs2):
        self.fig1s = figs1
        self.fig2s = figs2

    def get_spearman_rank_correlation(self):
        '''
        return the spearman rank correlation of the two figure collections
        '''

        n = self.fig1s.shape[0]
        fig1s = self.fig1s.reshape(n, -1)
        fig2s = self.fig2s.reshape(n, -1)
        m = fig1s.shape[1]
        fig1s_rank = [fig1s[i].argsort() + 1 for i in range(n)]
        fig2s_rank = [fig2s[i].argsort() + 1 for i in range(n)]
        fig1s_rank = np.array(fig1s_rank)
        fig2s_rank = np.array(fig2s_rank)
        ds = fig2s_rank - fig1s_rank
        ds = np.sum(np.power(ds, 2), axis=1)
        rhos = []
        for i in range(n):
            d = ds[i]

            rho = 1 - ((6 * d) / (m * m * m - m))
            rhos.append(rho)

        return rhos

    def get_ssim_similarity(self):
        """

        图片结构相似度

        SSIM一种衡量两幅图像结构相似度的新指标，其值越大越好，最大为1
        L这个参数，可能需要根据不同的任务图片改改
        """
        K1 = 0.01
        K2 = 0.03
        L = 1
        C1 = np.power(K1 * L, 2)
        C2 = np.power(K2 * L, 2)
        C3 = C2 / 2

        n = self.fig1s.shape[0]
        fig1s = self.fig1s.reshape(n, -1)
        fig2s = self.fig2s.reshape(n, -1)

        mu1s = np.mean(fig1s, axis=1)
        mu2s = np.mean(fig2s, axis=1)

        sigma1s = np.var(fig1s, axis=1)
        sigma2s = np.var(fig2s, axis=1)
        cov = [get_cov(fig1s[i], fig2s[i]) for i in range(n)]

        # print(np.array(cov).shape)

        l = [(2 * mu1s[i] * mu2s[i] + C1) / (mu1s[i] * mu1s[i] + mu2s[i] * mu2s[i] + C1) for i in range(n)]
        c = [(2 * sigma1s[i] * sigma2s[i] + C2) / (sigma1s[i] * sigma1s[i] + sigma2s[i] * sigma2s[i] + C2) for i in
             range(n)]
        s = [(cov[i] + C3) / (np.power(sigma1s[i] * sigma2s[i], .5) + C3) for i in range(n)]
        l = np.array(l)
        c = np.array(c)
        s = np.array(s)
        # print(l.shape, c.shape, s.shape)
        ssim = l * c * s
        ssim = np.abs(ssim)
        print(ssim.shape)
        for i in range(ssim.shape[0]):
            if ssim[i] > 1:
                ssim[i] = 1

        return ssim


def read_imgs(path):
    """
    read images from given folder
    :param path: the folder path
    :return: images, BHWC
    """
    dirs = os.listdir(path)
    imgs = []
    for fn in dirs:
        img_path = path + '/' + fn
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs


# Test
if __name__ == "__main__":
    path = "test_imgs"

    imgs = read_imgs(path)
    img1s = imgs[:2, :, :, :]
    img2s = imgs[2:, :, :, :]
    # 使用方法
    simi = similarity(img1s, img2s)
    ssim = simi.get_ssim_similarity()
    spearman = simi.get_spearman_rank_correlation()
    print("ssim", ssim)
    print("spearman", spearman)
