from __future__ import print_function
import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_cosine_similarity
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print("\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]))


def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getSelfSimilarity(input_m):
    # shape : N*256*256*3 => N*(256*256*3)
    input_m = input_m.view([input_m.shape[0], -1]).squeeze()
    # F.pdist => 1*(_nC_2) //
    res = pairwise_cosine_similarity(input_m)  # N*N
    # res = res.view([-1])
    return res


def getSelfCrossSimilarity(ss1, ss2):

    cosine = torch.nn.CosineSimilarity()
    SelfCrossSimilarity = cosine(ss1, ss2)
    SelfCrossSimilarity = SelfCrossSimilarity.view([-1])

    return SelfCrossSimilarity


def getIndex(size, index_A, index_B):
    """
      w = size * index_A + index_B - (index_A + 1) * (index_A + 2) * 0.5
      w = torch.as_tensor(w, dtype=torch.int64)
    """
    w = (size * index_A) + index_B
    w = torch.as_tensor(w, dtype=torch.int64)
    return w


def subSelfCrossSimilarity(size, SelfCrossSimilarity, batchIndex):  # Batchsize = 75
    num = len(batchIndex)
    r_Index = []

    for i, index in enumerate(batchIndex):
        index_A = index
        for j, index_2 in enumerate(batchIndex):
            index_B = index_2
            r_Index.append(getIndex(size, index_A, index_B))
    # print('length : ', len(r_Index))

    r_Index = torch.tensor(r_Index).unsqueeze(axis=-1)
    r_Index = r_Index.view([-1]).squeeze()

    sub_SelfCrossSimilarity = torch.gather(SelfCrossSimilarity, dim=0, index=r_Index)

    return sub_SelfCrossSimilarity.cuda()

