import tsne
from tqdm import tqdm
from data_io import load_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-90, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    #ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar



def optimize_artists_dictionary(artists):
    #convert each song list to a dictionary of songs
    for a in artists.values():
        a.song_list = {a.song_list[i].id: a.song_list[i] for i in range(0, len(a.song_list))}
    return artists

def attach_tsne_to_art_dict(artists, X,y):
    pbar = tqdm(total=len(y))
    print("Attaching tsne coordinates to artist dictionary")

    for i, row_label in enumerate(y):
        artists[row_label[0]].song_list[row_label[1]].tsne = [X[i][0],X[i][1]]
        pbar.update()
    pbar.close()
    return artists

def gen_heatmaps(artists, dimension, max, min):
    #given the dictionary of artists
    #retrive a tsne heatmap for each one of them
    # return the modified dictionary
    print('Generating artist heatmaps')
    pbar = tqdm(total=len(artists))
    for a in artists.values():
        a.tsne_heatmap = np.zeros((dimension,dimension))
        for s in a.song_list.values():
            row_idx = int(((s.tsne[0]+abs(min))/(max+abs(min))) * dimension)
            col_idx = int(((s.tsne[1]+abs(min))/(max+abs(min))) * dimension)
            a.tsne_heatmap[row_idx, col_idx] += 1

        #normalize by number of artists song
        a.tsne_heatmap /= len(a.song_list)
        pbar.update()
    pbar.close()
    return artists

def plot_heatmaps(artists,dimension, min, max):

    range = np.zeros((dimension))
    step = (max-min) / dimension
    for i,n in enumerate(range):
        left = min+i*step
        right = min+(i+1)*step
        range[i]=(right+left)/2
    for a in artists.values():

        fig, ax = plt.subplots()

        im, cbar = heatmap(a.tsne_heatmap, range, range, ax=ax,
                           cmap="viridis", cbarlabel="songs concentration")












        # We want to show all ticks...
        #ax.set_xticks(np.arange(dimension))
        #ax.set_yticks(np.arange(dimension))
        # ... and label them with the respective list entries
        #ax.set_xticklabels(farmers)
        #ax.set_yticklabels(vegetables)

        # Rotate the tick labels and set their alignment.
        #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        #for i in range(len(vegetables)):
        #    for j in range(len(farmers)):
        #        text = ax.text(j, i, harvest[i, j],
        #                       ha="center", va="center", color="w")
        title = "TSNE Heatmap for "+ a.name
        filename ='./Heatmaps/'+a.id
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')

def compute_heatmap_distance(h1,h2,dimension,metric):
    total_d = 0
    total_div = 0
    for i in range(dimension):
        for j in range(dimension):
            if metric == 'minkowski_2':
                d = abs(h1[i][j]-h2[i][j])
                total_d += d
            if metric == 'soergel_7':
                d = abs(h1[i][j]-h2[i][j])
                div = max(h1[i][j],h2[i][j])
                total_d += d
                total_div += div
            if metric == 'not_intersection_11':
                d = min(h1[i][j],h2[i][j])
                total_d += d

    if metric == 'soergel_7':
        total_d /= total_div
    if metric == 'not_intersection_11':
        total_d = 1 - total_d
    return total_d

def compute_distances(artists, dimension = 20 ,metric='minkowski_2'):
    distance_dict=dict()
    distance_mat=[]
    print('Computing distances between heatmaps using ',metric, ' metric...')
    pbar = tqdm(total=len(artists))
    for a_outer in artists.values():
        distance_dict[a_outer.id] = dict()
        for a_inner in artists.values():
            if a_inner.id not in distance_dict:
                dist = compute_heatmap_distance(h1=a_outer.tsne_heatmap,h2=a_inner.tsne_heatmap,dimension=dimension,metric=metric)
                distance_dict[a_outer.id][a_inner.id] = dist
                distance_mat.append([a_outer.id,a_inner.id,dist])
        pbar.update()
    pbar.close()

    return distance_dict,distance_mat


def main():
    artists = load_data(filename='full_msd_top20000.pkl')
    artists = tsne.filter_by_songlist_lenght(artists=artists, max_artists_num=20, min_lenght=0)
    X, y = tsne.prepare_dataset(artists)
    X = tsne.tsne(X,n_comp=2)
    artists = optimize_artists_dictionary(artists)
    artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)
    dim = 20
    artists = gen_heatmaps(artists=artists,dimension=dim, min=-85, max=85)
    plot_heatmaps(artists=artists,dimension=dim,min=-85, max=85)
    distance_dict,distance_mat = compute_distances(artists=artists,dimension=dim,metric='minkowski_2')
    distance_mat = np.array(distance_mat)

    distance_dict, distance_mat = compute_distances(artists=artists, dimension=dim, metric='soergel_7')
    distance_mat = np.array(distance_mat)

    distance_dict, distance_mat = compute_distances(artists=artists, dimension=dim, metric='not_intersection_11')
    distance_mat = np.array(distance_mat)

    return distance_dict, distance_mat
if __name__ == '__main__':
    main()