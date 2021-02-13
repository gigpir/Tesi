from data_io import load_data
from sklearn.manifold import TSNE
import numpy as np
import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from utility import resize_matrix, z_normalize, power_transform, normalize, quantile_transform, robust_scaler, \
    generate_color_text_list, gen_colors
from ellipses_plot_wrapper import confidence_ellipse
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_classification
from termcolor import colored
import matplotlib._color_data as mcd
def tsne(X, n_comp = 2):
    X_embedded = TSNE(n_components=n_comp,learning_rate=400,n_jobs=-1,random_state=12).fit_transform(X)# init = 'pca',
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

def tsne_plot_elliptical(X,y,artists,colors,filename='tsne_centroids',note='with outliers'):
    fig, ax_nstd = plt.subplots(figsize=(6, 6))

    out = []  # [<A.ID><A.C1><A.C2><v11><v22><v12>]
    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((X, y))

    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    #colors = generate_color_text_list(len(artist_ids))
    #colors = gen_colors(len(artist_ids))
    for i, id in enumerate(artist_ids):
        # filter only coordinates related to that id
        filtered_x = X_y[np.where(X_y[:, 2] == id)][:, :2].astype(np.float)

        x = filtered_x[:, 0]
        y = filtered_x[:, 1]
        ax_nstd.scatter(x, y, s=0.5, c=colors[i])
        confidence_ellipse(x, y, ax_nstd, n_std=1, label=r'$1\sigma$', edgecolor=colors[i])
        ax_nstd.scatter(np.mean(x), np.mean(y), c=colors[i], s=3)
        ax_nstd.annotate(artists[id].name, (np.mean(x), np.mean(y)),color=colors[i])

    fname = './plots/'+ filename+note+'.png'
    ax_nstd.set_title('TSNE-space centroids '+note)
    # ax.set_xlim([min(X[:, 0]), max(X[:, 0])])
    ax_nstd.set_xlim([-60, 60])
    ax_nstd.set_xlabel('tsne-1')
    # ax.set_ylim([min(X[:, 1]), max(X[:, 1])])
    ax_nstd.set_ylim([-50, 60])
    ax_nstd.set_ylabel('tsne-2')
    #ax_nstd.legend()
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

def get_features_dict():
    output = {0: 'mfcc_mean_0',
              1: 'mfcc_mean_1',
              2: 'mfcc_mean_2',
              3: 'mfcc_mean_3',
              4: 'mfcc_mean_4',
              5: 'mfcc_mean_5',
              6: 'mfcc_mean_6',
              7: 'mfcc_mean_7',
              8: 'mfcc_mean_8',
              9: 'mfcc_mean_9',
              10: 'mfcc_mean_10',
              11: 'mfcc_mean_11',
              12: 'pitch_mean_0',
              13: 'pitch_mean_1',
              14: 'pitch_mean_2',
              15: 'pitch_mean_3',
              16: 'pitch_mean_4',
              17: 'pitch_mean_5',
              18: 'pitch_mean_6',
              19: 'pitch_mean_7',
              20: 'pitch_mean_8',
              21: 'pitch_mean_9',
              22: 'pitch_mean_10',
              23: 'pitch_mean_11',

              24: 'loudness',
              25: 'tempo',

              26: 'mfcc_min_0',
              27: 'mfcc_max_0',
              28: 'mfcc_var_0',
              29: 'mfcc_min_1',
              30: 'mfcc_max_1',
              31: 'mfcc_var_1',
              32: 'mfcc_min_2',
              33: 'mfcc_max_2',
              34: 'mfcc_var_2',
              35: 'mfcc_min_3',
              36: 'mfcc_max_3',
              37: 'mfcc_var_3',
              38: 'mfcc_min_4',
              39: 'mfcc_max_4',
              40: 'mfcc_var_4',
              41: 'mfcc_min_5',
              42: 'mfcc_max_5',
              43: 'mfcc_var_5',
              44: 'mfcc_min_6',
              45: 'mfcc_max_6',
              46: 'mfcc_var_6',
              47: 'mfcc_min_7',
              48: 'mfcc_max_7',
              49: 'mfcc_var_7',
              50: 'mfcc_min_8',
              51: 'mfcc_max_8',
              52: 'mfcc_var_8',
              53: 'mfcc_min_9',
              54: 'mfcc_max_9',
              55: 'mfcc_var_9',
              56: 'mfcc_min_10',
              57: 'mfcc_max_10',
              58: 'mfcc_var_10',
              59: 'mfcc_min_11',
              60: 'mfcc_max_11',
              61: 'mfcc_var_11',

              62: 'pitch_min_0',
              63: 'pitch_max_0',
              64: 'pitch_var_0',
              65: 'pitch_min_1',
              66: 'pitch_max_1',
              67: 'pitch_var_1',
              68: 'pitch_min_2',
              69: 'pitch_max_2',
              70: 'pitch_var_2',
              71: 'pitch_min_3',
              72: 'pitch_max_3',
              73: 'pitch_var_3',
              74: 'pitch_min_4',
              75: 'pitch_max_4',
              76: 'pitch_var_4',
              77: 'pitch_min_5',
              78: 'pitch_max_5',
              79: 'pitch_var_5',
              80: 'pitch_min_6',
              81: 'pitch_max_6',
              82: 'pitch_var_6',
              83: 'pitch_min_7',
              84: 'pitch_max_7',
              85: 'pitch_var_7',
              86: 'pitch_min_8',
              87: 'pitch_max_8',
              88: 'pitch_var_8',
              89: 'pitch_min_9',
              90: 'pitch_max_9',
              91: 'pitch_var_9',
              92: 'pitch_min_10',
              93: 'pitch_max_10',
              94: 'pitch_var_10',
              95: 'pitch_min_11',
              96: 'pitch_max_11',
              97: 'pitch_var_11',

              98: 'mfcc_1st_0',
              99: 'mfcc_1st_1',
              100: 'mfcc_1st_2',
              101: 'mfcc_1st_3',
              102: 'mfcc_1st_4',
              103: 'mfcc_1st_5',
              104: 'mfcc_1st_6',
              105: 'mfcc_1st_7',
              106: 'mfcc_1st_8',
              107: 'mfcc_1st_9',
              108: 'mfcc_1st_10',
              109: 'mfcc_1st_11',
              110: 'pitch_1st_0',
              111: 'pitch_1st_1',
              112: 'pitch_1st_2',
              113: 'pitch_1st_3',
              114: 'pitch_1st_4',
              115: 'pitch_1st_5',
              116: 'pitch_1st_6',
              117: 'pitch_1st_7',
              118: 'pitch_1st_8',
              119: 'pitch_1st_9',
              120: 'pitch_1st_10',
              121: 'pitch_1st_11',

              122: 'mfcc_2nd_0',
              123: 'mfcc_2nd_1',
              124: 'mfcc_2nd_2',
              125: 'mfcc_2nd_3',
              126: 'mfcc_2nd_4',
              127: 'mfcc_2nd_5',
              128: 'mfcc_2nd_6',
              129: 'mfcc_2nd_7',
              130: 'mfcc_2nd_8',
              131: 'mfcc_2nd_9',
              132: 'mfcc_2nd_10',
              133: 'mfcc_2nd_11',
              134: 'pitch_2nd_0',
              135: 'pitch_2nd_1',
              136: 'pitch_2nd_2',
              137: 'pitch_2nd_3',
              138: 'pitch_2nd_4',
              139: 'pitch_2nd_5',
              140: 'pitch_2nd_6',
              141: 'pitch_2nd_7',
              142: 'pitch_2nd_8',
              143: 'pitch_2nd_9',
              144: 'pitch_2nd_10',
              145: 'pitch_2nd_11',
              }
    return output

def prepare_dataset(artists,remove_outliers=False, mode=0):

    """
            Extract information from data and build a dataset of the type X, y
            The features are standartized

            Parameters
            ----------
            artists : dict(Artist_id, Artist_obj)

            remove_outliers : Bool
                Perform outlier remotion or not {default = False}
            mode : int
                select what information to include in X {default = 0}

                0 : mfcc mean values (12), picht mean values (12),

                1 : mfcc mean values (12), pitch mean values (12)
                    mfcc min values (12), pitch min values (12),
                    mfcc max values (12), pitch max values (12),
                    mfcc var values (12), pitch var values (12),
                    s.tempo, s.loudness

                2 : mfcc mean values (12), pitch mean values (12)
                    mfcc min values (12), pitch min values (12),
                    mfcc max values (12), pitch max values (12),
                    mfcc var values (12), pitch var values (12),
                    mfcc 1st_grad values (12), pitch 1st_grad values (12),
                    s.tempo, s.loudness

                3 : mfcc mean values (12), pitch mean values (12)
                    mfcc min values (12), pitch min values (12),
                    mfcc max values (12), pitch max values (12),
                    mfcc var values (12), pitch var values (12),
                    mfcc 1st_grad values (12), pitch 1st_grad values (12),
                    mfcc 1nd_grad values (12), pitch 2nd_grad values (12),
                    s.tempo, s.loudness

            Output
            ---------
            X , y

    """


    rows = 1 #mean_n_rows(artists)
    pbar = tqdm(total=len(artists))

    #data will be in the format [feature_vector, ARTIST_ID, SONG_ID]
    X = []
    y = []
    for a in artists.values():
        for s in a.song_list:
            mfcc_mat = s.segments_timbre
            pitch_mat = s.segments_pitches

            if mode >= 0:
                feat_row = np.append(resize_matrix(mfcc_mat, rows), resize_matrix(pitch_mat, rows))
                feat_row = np.append(feat_row, [s.tempo, s.loudness])
            if mode >= 1:
                # append min, max, variance of each coloumn
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, min_max_var=True))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, min_max_var=True))
            if mode >= 2:
                # append first  derivative
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=1))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=1))
            if mode >= 3:
                #append second derivative
                feat_row = np.append(feat_row, resize_matrix(mfcc_mat, rows, gradient=2))
                feat_row = np.append(feat_row, resize_matrix(pitch_mat, rows, gradient=2))


            X.append(feat_row)
            lab_row = [a.id, s.id]

            #include also the first genre term if present
            if len(a.terms) > 0:
                lab_row.append(a.terms[0])
            else:
                lab_row.append('NULL')
            y.append(lab_row)
        pbar.update()
    pbar.close()
    #X = feature_selection(np.array(X).astype(np.float),np.array(y)[:,0])
    if remove_outliers:
        X ,y = remove_outliers_lof(X,y)
    #X = z_normalize(X)
    #X = power_transform(X)
    #X = quantile_transform(X)
    X = robust_scaler(X)

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


    for i in range(int(max_artists_num/2)):
        if len(sortedBySongNumber[i].song_list) >= min_lenght:
            new_artists[sortedBySongNumber[i].id] = sortedBySongNumber[i]
        else:
            break;
    offset = int(max_artists_num/2)
    left = max_artists_num - offset
    #add an equal number of artist based on artist similarity
    while left > 0:
        n_occ = 0
        for a in new_artists.values():
            if sortedBySongNumber[offset].id in a.similar_artists:
                n_occ += 1
        if n_occ > 1:
            new_artists[sortedBySongNumber[offset].id] = sortedBySongNumber[offset]
            left -= 1
        offset += 1
        if offset >= len(sortedBySongNumber):
            left = 0
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
def remove_outliers_znorm(X_tsne, y, n_sigma=2):
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

        # Simple Z distribution outlier detection
        z_mean_distances = (mean_distances[:, 1].astype(np.float) - mean_distances[:, 1].astype(np.float).mean()) / (mean_distances[:, 1].astype(np.float).std())
        for i, song_dist in enumerate(mean_distances):
            #leave about 95 percent in
            if z_mean_distances[i] > n_sigma:
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


def remove_outliers_lof(data, y):
    #Unsupervised Outlier Detection using Local Outlier Factor (LOF)
    out = []  # same format as X_tsne

    # get unique artist ids
    artist_ids = set(np.array(y)[:, 0])

    # aggregate horizontally X and y to filter
    X_y = np.hstack((data, y))

    black_list = []

    for id in artist_ids:
        # filter only coordinates related to that id
        filtered = X_y[np.where(X_y[:, -3] == id)]
        y = filtered[:, -3:]
        X = np.delete(filtered,np.s_[-3:],axis=1).astype(np.float)

        clf = LocalOutlierFactor(algorithm='auto',metric='euclidean',n_neighbors=10)
        pr = clf.fit_predict(X)
        for i, p in enumerate(pr):
            #if p == -1 we have an outlier
            if p == -1:
                black_list.append(y[i][1])
    X = []
    y = []
    for r in X_y:
        if r[-2] not in black_list:
            X.append(list(r[:-3]))
            y.append(list(r[-3:]))

    X = np.array(X).astype(np.float)
    y = np.array(y)
    #print("Outlier remotion: (%d - %d)= %d " % (X_y.shape[0], X_y.shape[0]-X.shape[0], X.shape[0]))
    print("Before Outlier remotion: ", X_y.shape)
    print("Before Outlier remotion: ", X.shape)

    return X,y



def create_dataframe_tsne(X,y):

    columns = ['tsne_1','tsne_2']
    data = pd.DataFrame(data=X, columns=columns)
    data['artist_id']= np.array(y)[:,0]
    data['song_id'] = np.array(y)[:,1]
    data['first_gen'] = np.array(y)[:,2]

    #data = data.astype({'tsne_1': np.float, 'tsne_2': np.float})
    return data

def compare_centroids_and_similar_artists(X_tsne, y, artists):
    centroids = get_centroids(X_tsne, y)

    distances = dict()
    for index1, outer in centroids.iterrows():
        # for each artist centroid calc the distance btw other centroids
        distances[outer['artist_id']] = []
        for index1, inner in centroids.iterrows():
            if outer['artist_id'] != inner['artist_id']:
                d = np.sqrt((outer['a_cen1']-inner['a_cen1'])**2+(outer['a_cen2']-inner['a_cen2'])**2)
                distances[outer['artist_id']].append([inner['artist_id'],d])

    #sort by ascending distance
    for key,val in distances.items():
        distances[key].sort(key=lambda tup: tup[1])

        #print(colored(distances[key],'green'))

        ground_t = artists[key].similar_artists
        new_gt=[]
        # consider only the artist present in our sample
        for id in ground_t:
            if id in distances:
                new_gt.append(id)

        print(artists[key].name, ' (', len(new_gt), ' occurrences )')
        if len(new_gt) != 0:
            for i, my_list in enumerate(distances[key]):
                if len(new_gt)>i:
                    print('\t\t', artists[my_list[0]].name,' --- ',artists[new_gt[i]].name)
                else:
                    print('\t\t', artists[my_list[0]].name)

def feature_selection(X,y):
    print('Before Feature selection ',X.shape)
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    min_features_to_select = 4  # Minimum number of features to consider
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy',
                  min_features_to_select=min_features_to_select,
                  n_jobs=4,
                  verbose=True)
    X_new = rfecv.fit_transform(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(min_features_to_select,
                   len(rfecv.grid_scores_) + min_features_to_select),
             rfecv.grid_scores_)
    plt.show()
    print('After Feature selection ', X_new.shape)

    return X_new



def main():

    artists = load_data(filename='full_msd_top20000.pkl')
    artists = filter_by_songlist_lenght(artists=artists, max_artists_num=20, min_lenght=0)

    colors = gen_colors(len(artists))

    X, y = prepare_dataset(artists)
    X_tsne = tsne(X,n_comp=2)
    tsne_plot_elliptical(X=X_tsne,y=y,artists=artists,colors=colors,filename='centroids_',note='with outliers')

    compare_centroids_and_similar_artists(X_tsne, y, artists)

    X, y = prepare_dataset(artists,remove_outliers=True)
    X_tsne = tsne(X, n_comp=2)
    tsne_plot_elliptical(X=X_tsne,y=y,artists=artists,colors=colors,filename='centroids_',note='without outliers')
    compare_centroids_and_similar_artists(X_tsne, y, artists)





if __name__ == '__main__':
    main()