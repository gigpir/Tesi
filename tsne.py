from data_io import load_data
from sklearn.manifold import TSNE
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from utility import resize_matrix, z_normalize


def tsne(X, n_comp = 2):
    X_embedded = TSNE(n_components=n_comp, init = 'pca',random_state=12).fit_transform(X)
    #manifold.TSNE(n_components=n_components, init='pca',random_state=0)
    return X_embedded
def tsne_plot_centroids(centroids,filename='tsne_centroids'):
    fig = plt.figure()
    y = np.array(centroids)[:, 0]
    X = np.array(centroids)[:, 1:6]
    X = X.astype(np.float)
    new_y = []
    ax = fig.add_subplot(1, 1, 1)

    vect = y
    y_dic = str_vect_to_dict(vect)
    for el in vect:
        new_y.append(y_dic[el])
    new_y = np.array(new_y)
    sizes = np.absolute(X[:, 4]) ** (1/2)
    ax.scatter(X[:, 0], X[:, 1], c=new_y, sizes=(sizes*200))
    ax.set_title('TSNE-centroids')
    pad = 20
    #ax.set_xlim([min(X[:, 0])-pad, max(X[:, 0])+pad])
    ax.set_xlim([-50, 60])
    ax.set_xlabel('tsne-1')
    #ax.set_ylim([min(X[:, 1])-pad, max(X[:, 1])+pad])
    ax.set_ylim([-50, 60])
    ax.set_ylabel('tsne-2')
    for i, txt in enumerate(vect):
        ax.annotate(np.array(centroids)[i, 6], (X[i, 0], X[i, 1]))
    fname = './plots/'+filename+'.png'
    plt.savefig(fname, dpi=600)
def tsne_plot(X,y,n_comp = 2, genre_annot=False,note=''):
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
        ax.scatter(X[:, 0], X[:, 1], c=new_y, s=10)
    else:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=new_y, s=10)

    ax.set_title('TSNE-space plot')
    #ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
    ax.set_xlim([-50, 60])
    ax.set_xlabel('tsne-1')
    #ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax.set_ylim([-50, 60])
    ax.set_ylabel('tsne-2')
    if n_comp == 3:
        ax.set_zlim([min(X[:, 2]), max(X[:, 2])])
        ax.set_zlabel('tsne-3')
    if genre_annot:
        for i, txt in enumerate(vect):
            if i % 100 == 0:
                ax.annotate(txt, (X[i, 0], X[i, 1]))
        fname = './plots/tsne_'+str(n_comp)+note+'_genre.png'
    else:
        fname = './plots/tsne_' + str(n_comp) +note+ '.png'

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

            #append first and second derivative
            #feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=1))
            #feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=1))
            #feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=2))
            #feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=2))

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

def get_centroids(X,y):

    out = [] # [<A.ID><A.C1><A.C2><v11><v22><v12>]
    #get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    #aggregate horizontally X and y to filter
    X_y = np.hstack((X, y))

    for id in artist_ids:
        #filter only coordinates related to that id
        filtered_x = X_y[np.where(X_y[:, 2] == id)][:, :2].astype(np.float)
        #filtered_x = df.loc[['artist_id'] == id]
        genre = X_y[np.where(X_y[:, 2] == id)][0, 4]
        # calc centroid on that artist
        c1 = np.mean(filtered_x[:, 0])
        c2 = np.mean(filtered_x[:, 1])
        var = np.cov(filtered_x[:, 0], filtered_x[:, 1])
        out.append([id, c1, c2, var[0][0], var[1][1], var[0][1], genre])

    out = pd.DataFrame(data=out, columns=["artist_id", "a_cen1", "a_cen2", "a_var11", "a_var22", "a_var12", "first_gen"])
    out = out.astype({'a_cen1': np.float, 'a_cen2': np.float, 'a_var11': np.float, 'a_var22': np.float, 'a_var12': np.float})

    return out
def remove_outliers(X_tsne,y, n_sigma=2):
    out = []  # same format as X_tsne

    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((X_tsne, y))

    black_list = []

    for id in artist_ids:
        # filter only coordinates related to that id
        filtered = X_y[np.where(X_y[:, 2] == id)]
        X = filtered[:,:2].astype(np.float)
        y = filtered[:, 3]
        # for every point retreive its mean distance respect to all other points from the
        # same label
        mean_distances = []
        for i, point1 in enumerate(X):
            song_id = y[i]
            distances = []
            for j, point2 in enumerate(X):
                if i != j:
                    d = abs(point1[0]-point2[0])+abs(point1[1]-point2[1])
                    distances.append(d)
            mean_dist = np.mean(np.array(distances))
            mean_distances.append([song_id, mean_dist])

        mean_distances = np.array(mean_distances)
        z_mean_distances = (mean_distances[:, 1].astype(np.float) - mean_distances[:, 1].astype(np.float).mean()) / (mean_distances[:, 1].astype(np.float).std())
        for i, song_dist in enumerate(mean_distances):
            #leave about 95 percent in
            if z_mean_distances[i] > n_sigma or z_mean_distances[i] < -n_sigma:
                black_list.append(song_dist[0])

    X = []
    y = []
    for r in X_y:
        if r[3] not in black_list:
            X.append(r[0:2])
            y.append(r[2:5])
    X = np.array(X).astype(float)
    y = np.array(y)
    print("Outlier remotion: (%d - %d)= %d " % (X_y.shape[0], X_y.shape[0]-X.shape[0], X.shape[0]))



    return X,y

def create_dataframe_tsne(X,y):

    columns = ['tsne_1','tsne_2']
    data = pd.DataFrame(data=X, columns=columns)
    data['artist_id']= np.array(y)[:,0]
    data['song_id'] = np.array(y)[:,1]
    data['first_gen'] = np.array(y)[:,2]

    #data = data.astype({'tsne_1': np.float, 'tsne_2': np.float})
    return data



def main():

    artists = load_data(filename='full_msd_top20000.pkl')
    artists = filter_by_songlist_lenght(artists=artists, max_artists_num=10, min_lenght=0)
    X, y = prepare_dataset(artists)
    X_tsne = tsne(X,n_comp=2)

    centroids = get_centroids(X_tsne, y)
    tsne_plot_centroids(centroids=centroids,filename='centroids_with_outliers')
    tsne_plot(X=X_tsne, y=y, n_comp=2, genre_annot=True, note='_with_outliers')

    X_tsne,y = remove_outliers(X_tsne,y)

    centroids = get_centroids(X_tsne, y)
    tsne_plot_centroids(centroids=centroids, filename='centroids_wo_outliers')
    tsne_plot(X=X_tsne, y=y, n_comp=2, genre_annot=True, note='_wo_outliers')



if __name__ == '__main__':
    main()