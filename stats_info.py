import numpy as np
from matplotlib import pyplot as plt


def stats_artists_songs(artists):
    #given a dictionary<artist.id><artist>
    #print information about number of songs
    avg_songs=0
    d = dict()
    for a in artists:
        avg_songs += len(artists[a].song_list)
        if len(artists[a].song_list) not in d:
            d[len(artists[a].song_list)] = 1
        else:
            d[len(artists[a].song_list)] += 1
    try:
        avg_songs /= len(artists.keys())
    except ZeroDivisionError:
        avg_songs = 0


    var = 0
    for a in artists:
        var += (len(artists[a].song_list)-avg_songs)**2
    try:
        var /= len(artists.keys())
    except ZeroDivisionError:
        var = 0
    y = np.array(list(d.values()))
    x = np.array(list(d.keys()))
    # plot the data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(x, y, zorder=1, c='c', s=1, alpha=0.1)
    ax.set_title('Distribution of Number of songs per artist')
    ax.set_xlim([0, max(x)])
    ax.set_xlabel('N of songs')
    ax.set_ylim([0, max(y)])
    ax.set_ylabel('Count')

    fname = "./stats_plot/songsNumber.png"
    plt.savefig(fname, dpi=400)
    print("avg number of song per artist = %f\nVariance = %f\n" % (avg_songs,var))