import tsne
from tqdm import tqdm
from data_io import load_data, save_data
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd
import rbo
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

    """
        convert each song list to a dictionary of songs
        this is useful for speeding the search later
    """

    for a in artists.values():
        a.song_list = {a.song_list[i].id: a.song_list[i] for i in range(0, len(a.song_list))}
    return artists

def attach_tsne_to_art_dict(artists, X,y):
    """
        each song of each Artist will have its tsne coordinates associated
    """
    print("Attaching tsne coordinates to artist dictionary")
    pbar = tqdm(total=len(y))
    for i, row_label in enumerate(y):
        artists[row_label[0]].song_list[row_label[1]].tsne = [X[i][0],X[i][1]]
        pbar.update()
    pbar.close()
    return artists

def gen_heatmaps(artists, dimension, max, min):
    """
        given the dictionary of artists
        retrive a tsne heatmap for each one of them
        return the modified dictionary
    """

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

        title = "TSNE Heatmap for "+ a.name
        filename ='./Heatmaps/'+a.id
        ax.set_title(title)
        fig.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close('all')

def compute_heatmap_distance(h1,h2,dimension,metric):
    """
        Parameters
        ----------
        h1 : 2d array
            The name of the animal
        h2 : 2d array
            The sound the animal makes
        dimension : int
            array dimension
        metric : str
            [minkowski_2, soergel_7, not_intersection_11, kullback-leibler_37]
            see http://www.fisica.edu.uy/~cris/teaching/Cha_pdf_distances_2007.pdf for info

        Output
        ---------
        total_d : float
            the greater total_d is the farther h1 and h2 are
        """
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
            if metric == 'kullback-leibler_37':
                d = h1[i][j] * np.log(h1[i][j]/h2[i][j])
                total_d += d

    if metric == 'soergel_7':
        total_d /= total_div
    if metric == 'not_intersection_11':
        total_d = 1 - total_d
    return total_d

def compute_distances(artists, dimension = 20 ,metric='minkowski_2'):
    """
            Parameters
            ----------
            artists : dict of Artist object

            dimension : int
                desired dimension for heatmap {default = 20}
            metric : str
                [minkowski_2, soergel_7, not_intersection_11, kullback-leibler_37]
                see http://www.fisica.edu.uy/~cris/teaching/Cha_pdf_distances_2007.pdf for info

            Output
            ---------
            distance_dict : dict(artist_i_id, dict(artist_j_id, distance)

            distance_mat : list
                each row contains [Artist1_id,Artist2_id,distance]
    """
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

def compute_ranking(artists, distance_mat, top_k):
    """
    Compute ranking for each artist
                Parameters
                ----------
                artists : dict of Artist object

                distance_mat : list
                    each row contains [Artist1_id,Artist2_id,distance]

                top_k : int
                    lenght of ranked list
                Output
                ---------
                artist dictionary
        """
    for a in artists.values():
        ranked_list = []

        # isolate only the similarity points that belong to artist a
        slice = [row for row in distance_mat if (row[0] == a.id or row[1] == a.id)]

        # sort by increasing distance
        slice.sort(key=lambda row: row[2], reverse=False)

        # take only first TOP_N results
        slice = slice[:top_k]

        for s in slice:
            if s[0] != a.id:
                ranked_list.append(s[0])

            if s[1] != a.id:
                ranked_list.append(s[1])

        a.my_similar_artists = ranked_list

    return artists

def compute_ranking_score(artists, **kwargs):
    """
        Give a score to a ranking algorithm
                    Parameters
                    ----------
                    artists : dict of Artist object

                    **kwargs : list
                        e.g. metric = 'minkowsky'
                    Output
                    ---------
                    score : float
    """
    scores = []
    print('Computing ranking score my rank and ground truth...')
    for a in artists.values():
        if len(a.similar_artists) > 0:
            scores.append(rbo.RankingSimilarity(a.similar_artists, a.my_similar_artists).rbo())

    scores = np.array(scores)

    if len(kwargs) != 0:
        for key, val in kwargs.items():
            print('The Average score (RBO) for the metric ', val, ' is ', np.mean(scores))

def clean_similar_artists(artists):
    """
        Remove artists that are not present in the current
        selection
            Parameters
            ----------
            Output
            --------
            artists : dict of Artist object
    """
    zero_lenght_ranks = 0
    lenghts_array = []
    for a in artists.values():
        new_list = []
        for sim_a in a.similar_artists:
            if sim_a in artists:
                new_list.append(sim_a)
        a.similar_artists = new_list

        #just for statistical purposes
        if len(new_list) == 0:
            zero_lenght_ranks += 1
        else:
            lenghts_array.append(len(new_list))

    lenghts_array = np.array(lenghts_array)
    mean_lenght = np.mean(lenghts_array)

    print('Cleaning phase completed:')
    print(zero_lenght_ranks ,' artists have zero similar artists in their ground truth')
    print('Mean lenght of ranking array is (zero rank were not considered) ', mean_lenght)

    return artists

def check_if_ranked_list_are_equal(distance_mat):

    for mat in distance_mat.values():
        mat.sort(key=lambda row: row[2], reverse=False)

    mats = list(distance_mat.values())

    for i in range(len(mats[0])):
        if mats[0][i][0] != mats[1][i][0]:
            return False
        if mats[0][i][1] != mats[1][i][1]:
            return False
        if mats[0][i][0] != mats[2][i][0]:
            return False
        if mats[0][i][1] != mats[2][i][1]:
            return False
    return True

def main():
    dim = 20
    tmp = True
    metrics = ['minkowski_2', 'soergel_7', 'not_intersection_11']  # ,'kullback-leibler_37']
    if not tmp:
        artists = load_data(filename='full_msd_top20000.pkl')
        #artists = tsne.filter_by_songlist_lenght(artists=artists, max_artists_num=20, min_lenght=0)
        X, y = tsne.prepare_dataset(artists)
        X = tsne.tsne(X,n_comp=2)


        artists = optimize_artists_dictionary(artists)
        artists = attach_tsne_to_art_dict(artists=artists, X=X, y=y)

        artists = gen_heatmaps(artists=artists,dimension=dim, min=-85, max=85)
        plot_heatmaps(artists=artists,dimension=dim,min=-85, max=85)



        artists = clean_similar_artists(artists=artists)

        save_data(artists, filename='tmp.pkl')
    else:
        artists = load_data(filename='tmp.pkl')

    distance_dict = dict()
    distance_mat = dict()

    for metric in metrics:
        distance_dict[metric], distance_mat[metric] = compute_distances(artists=artists,dimension=dim,metric=metric)
        #distance_mat[metric] = np.array(distance_mat)
        filename = './Heatmaps/distance_'+metric+'.csv'
        pd.DataFrame(distance_mat[metric]).to_csv(filename)

        artists = compute_ranking(artists=artists,distance_mat=distance_mat[metric], top_k=20)
        compute_ranking_score(artists=artists,metric=metric)


    return distance_dict, distance_mat
if __name__ == '__main__':
    distance_dict, distance_mat = main()

    if check_if_ranked_list_are_equal(distance_mat=distance_mat):
        print("OK, equal")
    else:
        print("NOT equal")