import numpy as np
from sklearn.preprocessing import StandardScaler


def decrease_vector_size(vect,final_size):
    # Decrease array size by averaging adjacent values with numpy
    final_vect = []

    in_dim = vect.shape[0]
    window_size = int(in_dim / final_size)

    for i in range(final_size):
        idx_in = i * window_size
        idx_out = idx_in + window_size
        #calculate average of the group
        if i == final_size - 1:
            # last element, pick til the end
            final_vect.append(np.mean(vect[idx_in:]))
        else:
            final_vect.append(np.mean(vect[idx_in:idx_out]))
    return np.array(final_vect)


def increase_vector_size(vect,final_size):
    # Increase array size by duplicating each value in order to reach the target size
    in_dim = vect.shape[0]
    dup_factor = int(final_size/in_dim)

    for i in range(in_dim-1):
        v = np.full((dup_factor), vect[i])
        if i == 0:
            final_vect = v
        else:
            final_vect = np.append(final_vect, v)

    #create a last vector in order to reach the target size
    if final_size-final_vect.shape[0] != 0:
        v = np.full((final_size-final_vect.shape[0]), vect[in_dim-1])
        final_vect = np.append(final_vect, v)
    return final_vect


def resize_matrix(matrix, row):
    # reduce or increase the number of rows of matrix in order
    # to achieve dimension specified by row

    if matrix.shape[0]>row:
        resize_vect = decrease_vector_size
    else:
        resize_vect = increase_vector_size

    for i in range(matrix.shape[1]):
        #extract and transpose the corresponding coloumn vector
        new_vect = resize_vect(matrix[:,i],row)
        if i == 0:
            final = new_vect
        else:
            final = np.append(final,new_vect)
    return final


def z_normalize(data):
    return StandardScaler().fit(data).transform(data)