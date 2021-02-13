from data_io import load_data
import tsne


def feat_selection(X,y, n_initial_features):
    """
        each iteration a the most performing feature will be added
    """

    idx_name_map = tsne.get_features_dict()

    #for i in range(n_initial_features):




def main():
    """
        typical usage of feature selection

    """

    #load dataset
    artists = load_data(filename='full_msd_top20000.pkl')
    #reduce num of artists - optional
    artists = tsne.filter_by_songlist_lenght(artists=artists, max_artists_num=20, min_lenght=0)


if __name__ == '__main__':
    main()