import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.spatial import Voronoi
from io import BytesIO
import base64

def fig_to_uri(fig, close_all=True):
    out_img = BytesIO()
    fig.savefig(out_img, format='png')
    if close_all:
        fig.clf()
        plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

def compute_vectors(eulers, direction):
    
    def vector_project(vect):
        rZ = np.empty(vect.shape[0])
        theZ = np.empty(vect.shape[0])
        for k, vect in enumerate(vect):
            theZ[k] = np.arctan2(vect[1], vect[0])
            rZ[k]   = np.sin(np.arccos(vect[2]))/(1+np.arccos(np.cos(vect[2])))
        xes_preds = rZ*np.cos(theZ)
        yes_preds = rZ*np.sin(theZ)
        return xes_preds, yes_preds

    i1, i2, i3 = eulers.T
    i1c = np.cos(i1)
    i1s = np.sin(i1)
    i2c = np.cos(i2)
    i2s = np.sin(i2)
    i3c = np.cos(i3)
    i3s = np.sin(i3)
    x00 = i1c*i2c*i3c-i1s*i3s
    x01 = -i3c*i1s-i1c*i2c*i3s
    x02 = i1c*i2s
    x10 = i1c*i3s+i2c*i3c*i1s
    x11 = i1c*i3c-i2c*i1s*i3s
    x12 = i1s*i2s
    x20 = -i3c*i2s
    x21 = i2s*i3s
    x22 = i2c
    c0 = np.stack((x00,x01,x02), axis=1)
    c1 = np.stack((x10,x11,x12), axis=1)
    c2 = np.stack((x20,x21,x22), axis=1)
    rot_mat = np.stack((c0,c1,c2), axis=1)
    rot_mat_inv = np.linalg.inv(rot_mat)
    
    xv, yv, zv = list(map(lambda idx: np.sort(normalize(np.abs(idx), axis=1), axis=1), 
                          np.transpose(rot_mat_inv, [2, 0, 1])))
    
    if direction=='z':
        yes, xes = vector_project(zv)
    elif direction=='y':
        yes, xes = vector_project(yv)
    elif direction=='x':
        yes, xes = vector_project(xv)
    else:
        yes, xes = vector_project(zv)
    
    x_points = np.vstack((xes, yes)).T
    
    return x_points

def fill_plot_outline(ax):
    xcr = np.sqrt(2) * np.cos(np.radians(np.linspace(0, 45, 50))) - 1.0
    ycr = np.sqrt(2) * np.sin(np.radians(np.linspace(0, 45, 50)))
    xcr = np.append(xcr, [0.45, 0.45])
    ycr = np.append(ycr, [0.4,0.0])
    ax.fill([0.0,0.0,0.4], [0.0,0.4,0.4], 'black', zorder=10)
    ax.fill(xcr, ycr, 'black', zorder=10)
    
    # Add annotations
    ax.text(-0.02, -0.02, '100', color='white')
    ax.text(0.40, -0.02, '110', color='white')
    ax.text(0.354, 0.375, '111', color='white', zorder=11)
    return ax

def get_region_values(vor, x_points, z_values):
    knn_idx = np.argmin(pairwise_distances(X=x_points, Y=vor.points, metric='euclidean'), axis=1)
    region_values = np.zeros((len(vor.points)))
    n_vals_cell = np.zeros((len(vor.points)))
    for ind, zval in zip(knn_idx, z_values):
        n_vals_cell[ind] += 1
        region_values[ind] += (zval - region_values[ind]) / n_vals_cell[ind]
    return region_values

def set_colorbar(region_values, ax, vor, vmin, vmax, cmap):
    mapper = plt.cm.ScalarMappable(norm=clrs.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            plt.fill(*zip(*[vor.vertices[i] for i in region]), color=mapper.to_rgba(region_values[r]))
    cb = plt.colorbar(mapper)
    cb.ax.tick_params(colors='white')
    return ax

def get_voronoi(n_fib_tiles):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(n_fib_tiles*16):
        theta = phi * i        
        y = 1 - (i / float(n_fib_tiles*16 - 1)) * 2 
        x = np.cos(theta) * np.sqrt(1 - y * y)
        z = np.sin(theta) * np.sqrt(1 - y * y) 
        r = np.sqrt(x**2 + y**2 + z**2)
        sss = np.array([-x / (1-z), -y / (1-z)])
        if (np.linalg.norm(sss) <= 0.65) & (np.arccos(z / r) > np.radians(90)) & (x <= 0.1) & (y <= 0.1):
            points.append(sss)
    points = np.array(points)
    vor = Voronoi(points)
    return vor


if __name__ == '__main__':
    coords001 = compute_vectors(np.array([[0, 0, 0]]), 'z')
    coords011 = compute_vectors(np.radians(np.array([[0, 45, 0]])), 'z')
    coords111 = compute_vectors(np.radians(np.array([[0, 45, 36.5]])), 'z')
        
    import matplotlib.pyplot as plt
    u = np.vstack([coords001, coords011, coords111])
    plt.scatter(u[:,0], u[:,1])
    plt.show()
    