from data_io import load_data
from sklearn.manifold import TSNE
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

from utility import resize_matrix, z_normalize


def tsne(X, n_comp = 2):
    X_embedded = TSNE(n_components=n_comp, init = 'pca',random_state=12).fit_transform(X)
    #manifold.TSNE(n_components=n_components, init='pca',random_state=0)
    return X_embedded

def tsne_plot(X,y,n_comp = 2, genre_annot=False):
    fig = plt.figure()
    y = np.array(y)
    new_y = []

    if genre_annot:
        #each first song genre becomes an integer
        vect = y[:, 2]  # artist.terms[0] of each song
    else:
        # each distinct artist becomes an integer
        vect = y[:, 0]  # artist.id of each song

    y_dic = str_vect_to_dict(vect)
    for el in vect:
        new_y.append(y_dic[el])
    new_y = np.array(new_y)
    #new_y is an ordered array of integers, each integer correspond to an artist/genre

    if n_comp == 2:
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(X[:, 0], X[:, 1], c=new_y, s=50)
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=new_y, s=50)

    ax.set_title('TSNE-space plot')
    ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
    ax.set_xlabel('tsne-1')
    ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax.set_ylabel('tsne-2')
    if n_comp == 3:
        ax.set_zlim([min(X[:, 2]), max(X[:, 2])])
        ax.set_zlabel('tsne-3')
    if genre_annot:
        for i, txt in enumerate(vect):
            if i % 2 == 0:
                ax.annotate(txt, (X[i, 0], X[i, 1]))
        fname = './plots/tsne_'+str(n_comp)+'_genre.png'
    else:
        fname = './plots/tsne_' + str(n_comp) + '.png'

    plt.savefig(fname, dpi=600)


def mean_n_rows(artists):
    # retrieve the mean value of rows of each mfcc matrix
    mean = 0
    n = 0
    min = 9999999
    for a in artists.values():
        for s in a.song_list:
            print(s.segments_timbre.shape)
            mean += s.segments_timbre.shape[0]
            n += 1
            if s.segments_timbre.shape[0] < min:
                min = s.segments_timbre.shape[0]
    return mean


def prepare_dataset(artists):

    #normalize dimensions of mfcc matrix by decreasing or increasing feature space
    # normalize dimensions of pitches matrix by decreasing or increasing feature space
    #retrive the mean value of rows of each mfcc matrix, the number of cols is constant (12)

    rows = 1 #mean_n_rows(artists)
    pbar = tqdm(total=len(artists))

    #data will be in the format [feature_vector, ARTIST_ID, SONG_ID]
    X = []
    y = []
    for a in artists.values():
        for s in a.song_list:
            mfcc_mat = s.segments_timbre
            pitch_mat = s.segments_pitches
            feat_row = np.append(resize_matrix(mfcc_mat, rows), resize_matrix(pitch_mat, rows))
            feat_row = np.append(feat_row, [s.tempo, s.loudness])
            X.append(feat_row)
            lab_row = [a.id, s.id]

            #include also the first genre term if present
            if len(a.terms) > 0:
                lab_row.append(a.terms[0])
            else:
                lab_row.append('NULL')
            y.append(lab_row)
        pbar.update()

    X = z_normalize(X)
    return X, y

def str_vect_to_dict(vect):
    #given a list of string
    #return a dict where the key is a string anc value is an int
    y = dict()
    n = 0
    for str in vect:
        if str not in y:
            y[str] = n
            n += 1
    return y

def filter_by_songlist_lenght(artists, max_artists_num=100, min_lenght = 10):
    #given a dictionary of artists
    #return a dictionary of at maximum max_artists_num ordered elements
    # each one of these elements has at least min_lenght songs associated
    new_artists = dict()
    sortedBySongNumber = sorted(artists.values(), reverse=True, key=lambda x: len(x.song_list))

    for i in range(max_artists_num):
        if len(sortedBySongNumber[i].song_list) >= min_lenght:
            new_artists[sortedBySongNumber[i].id] = sortedBySongNumber[i]
        else:
            break;
    return new_artists



def main():

    artists = load_data(filename='data_subset.pkl')
    artists = filter_by_songlist_lenght(artists=artists, max_artists_num=10, min_lenght=0)
    X, y = prepare_dataset(artists)
    X_tsne = tsne(X,n_comp=2)
    tsne_plot(X=X_tsne, y=y, n_comp=2, genre_annot=True)
    tsne_plot(X=X_tsne, y=y, n_comp=2, genre_annot=False)
    X_tsne = tsne(X, n_comp=3)
    tsne_plot(X=X_tsne, y=y, n_comp=3, genre_annot=False)

    return X_tsne, y


if __name__ == '__main__':
    X_tsne,y = main()