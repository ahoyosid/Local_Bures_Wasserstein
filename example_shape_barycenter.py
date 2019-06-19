import numpy as np
import torch
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from skimage import io
from sklearn.utils import check_random_state
from skimage.transform import resize
from sklearn.mixture import GaussianMixture
from utils import generate_samples
from utils import plot_ellipses_and_assignment
from approximated_transport import (transport_samples_to_barycenter,
                                    get_assignment,
                                    get_pairwise_barycenter)

figsize = (8, 8)
current_palette = sns.color_palette()
rnd = check_random_state(seed=3)


### Load Images
n_x = n_y = 200

## Cat
image1_ = (1 - io.imread("data/shapes/cat_1.jpg")[:, :, 0] / 255)
image1 = resize(image1_, (200, 200), mode="reflect", anti_aliasing=False).astype("bool") * 1
## Rabbit
image2_ = (1 - io.imread("data/shapes/rabbit.png")[:, :, 1] / 255).astype("bool")
image2 = resize(image2_, (image1.shape[0], image1.shape[1]), mode="reflect", anti_aliasing=False).astype("bool") * 2

image = np.zeros((600, 600))
image[30:30 + n_x, 10:10 + n_y] = image1
image[320:320 + n_x, 350:350 + n_y] = image2[:, ::-1]

data = generate_samples(image, n_y, n_samples=60000, random_state=43)
X = data[0]
Y = data[1]

### Fit GMMs
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, random_state=54)
Y_train, Y_test = train_test_split(Y, random_state=54)

n_components = 20

gmm_x = GaussianMixture(n_components=n_components,
                        random_state=3).fit(X_train)
gmm_y = GaussianMixture(n_components=n_components,
                        random_state=34).fit(Y_train)

assignment_xy = get_assignment(gmm_x, gmm_y)

#### Save figure
plot_ellipses_and_assignment(X_train, Y_train, gmm_x, gmm_y, assignment_xy)


### Various barycenters
plt.close("all")

lambds = [
    np.array([.9, .1]),
    np.array([.75, .25]),
    np.array([.5, .5]),
    np.array([.25, .75]),
    np.array([.1, .9]),
]

fig, axx = plt.subplots(1, 5, figsize=[15, 3])

for ind, l in enumerate(lambds):

    means_k, covs_k = get_pairwise_barycenter(
        gmm_x, gmm_y, assignment_xy, lambds=l)

    X_trans, Y_trans = transport_samples_to_barycenter(
        X_train, Y_train, gmm_x, gmm_y,
        assignment_xy, means_k, covs_k)

    ## Original samples
    axx[ind].scatter(*X_train.T, c=current_palette[0], s=1)
    axx[ind].scatter(*Y_train.T, c=current_palette[1], s=1)
    ## Transported samples
    axx[ind].scatter(*X_trans.T, c=current_palette[5], s=1, alpha=0.1)
    axx[ind].scatter(*Y_trans.T, c=current_palette[5], s=1, alpha=0.1)

    axx[ind].set_xticks([])
    axx[ind].set_yticks([])
    axx[ind].set_title("Approximated barycenter\n" + r"$\lambda=$%s" % l)
    axx[ind].set_aspect('equal', 'datalim')

    sns.despine(left=True, bottom=True)

plt.subplots_adjust(hspace=0.05, wspace=0.08)
plt.savefig("example_interpolation.png", bbox_inches='tight')
