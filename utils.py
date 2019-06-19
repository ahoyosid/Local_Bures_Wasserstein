import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve  
from sklearn.utils import check_random_state


def generate_samples(image, n_y, n_samples=1000, random_state=43):
    """ Generate samples given the support.
    
    image: Int-like image.
    
    n_samples: number of samples to draw from the whole image.
    
    random_state: Seed.
    
    Returns:
    --------
    
    data: list of samples, (n_images, n_samples_image)  
        
    """
    rnd = check_random_state(seed=random_state)
    x_ini = rnd.randint(low=0, high=image.shape[0], size=n_samples)
    y_ini = rnd.randint(low=0, high=image.shape[1], size=n_samples)

    # Creates binary mask
    mask = np.zeros_like(image)
    for i, j in zip(x_ini, y_ini):
        mask[i, j] = 1
        
    data = []
    elements = np.unique(image)
    for element in elements:
        if element != 0:
            img = mask * (image == element)
            y1, x1  = np.where(img == 1)
            x_a, y_a = x1, n_y - y1

            data.append(np.hstack([x_a[:, np.newaxis], y_a[:, np.newaxis]]))
    return data


def make_ellipses(means, covariances, ax, color, alpha=0.5):
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(means, v[0], v[1],
                              180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(alpha)
    ax.add_artist(ell)
    ax.set_aspect('equal', 'datalim')
    pass


def plot_assignment(X, Y, gmm_x, gmm_y, assignment_xy, 
                    ax=None, figsize=[5, 5], 
                    num_max=100, alpha_step=0.15, 
                    current_palette=sns.color_palette()):
    xs = gmm_x.means_
    xt = gmm_y.means_
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    G = np.vstack(assignment_xy).T
    ind = 0
    alpha = 1.    
    for i in range(xs.shape[0]):
        for j in range(xt.shape[0]):
            if G[i, 1] == j:
                if (ind < num_max):
                    ax.plot([xs[i, 0], xt[j, 0]], [xs[i, 1], xt[j, 1]], 
                            c="k", alpha=alpha)
                    ind += 1
                    alpha *= (1 - alpha_step)

    ax.scatter(*X.T, c=current_palette[0], s=10)
    ax.scatter(*Y.T, c=current_palette[1], s=10)
    ax.scatter(*np.vstack(gmm_x.means_).T, c="k", s=10)
    ax.scatter(*np.vstack(gmm_y.means_).T, c="k", s=10)    

    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(left=True, bottom=True)
    ax.set_aspect('equal', 'datalim')
    pass



def plot_ellipses_and_assignment(X, Y, gmm_x, gmm_y, assignment_xy, 
                                 current_palette=sns.color_palette(), 
                                 fig_name="assignment.png"):

    plt.close("all")
    _, axx = plt.subplots(1, 2, figsize=[12, 4])

    axx[0].scatter(*X.T, color=current_palette[0], s=1)
    axx[0].scatter(*Y.T, color=current_palette[1], s=1)

    for i in range(gmm_x.n_components):
        make_ellipses(gmm_x.means_[i], gmm_x.covariances_[i], axx[0], "brown")
    for i in range(gmm_y.n_components):
        make_ellipses(gmm_y.means_[i], gmm_y.covariances_[i], axx[0], "brown")

    axx[0].scatter(*np.vstack(gmm_x.means_).T, c="k", s=10)
    axx[0].scatter(*np.vstack(gmm_y.means_).T, c="k", s=10) 

    axx[0].set_xticks([])
    axx[0].set_yticks([])

    plot_assignment(X, Y, gmm_x, gmm_y, 
                    assignment_xy, ax=axx[1])

    sns.despine(left=True, bottom=True)
    plt.savefig(fig_name, bbox_inches='tight')

    pass



def roc_plot(y_true, y_pred, z, ax, fontsize=16):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    sfpr, stpr, _ = roc_curve(z, y_pred)
    
    roc = np.vstack([tpr, fpr])
    sroc = np.vstack([stpr, sfpr])
    
    ax.plot(*roc, color="m")
    ax.plot(*sroc, color="b")
        
    ax.set_xlabel("FPR", fontsize=fontsize)
    ax.set_ylabel("TPR", fontsize=fontsize)
    ax.fill_between(sroc[0], sroc[0], sroc[1], 
                    facecolor='b', alpha=0.3)
    
    ax.plot([0, 1], [0, 1], "k")
    ax.legend(fontsize=fontsize)