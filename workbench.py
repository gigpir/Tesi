from data_io import load_data,retrieve_artist_dict,save_data, traverse_exp,create_filter
from stats_info import stats_artists_songs


#artists = retrieve_artist_dict()
#save_data(artists)

#del artists

#arts = load_data()
#stats_artists_songs(arts)

#exp_dict = traverse_exp()
#save_data(exp_dict, filename='dict_art_nsongs.pkl')
'''
filter = create_filter(pkl_dict_filename='dict_art_nsongs.pkl', max_songs=1500)
n = 0
for key,value in filter.items():
    print(key, value)
    n+=value
nn=0

artists_filtered = retrieve_artist_dict(basedir='./millionsongsubset_full/MillionSongSubset/data', filter=filter)

for key, a in artists_filtered.items():
    if key not in filter:
        print('ERROR')
        break
    nn += len(a.song_list)

print(n, nn)
'''



