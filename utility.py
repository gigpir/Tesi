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


def resize_matrix(matrix, row, gradient=0, min_max_var=False):
    # reduce or increase the number of rows of matrix in order
    # to achieve dimension specified by row

    if matrix.shape[0]>row:
        resize_vect = decrease_vector_size
    else:
        resize_vect = increase_vector_size

    for i in range(matrix.shape[1]):
        if min_max_var:
            # extract min max and variance of coloumn vector
            if i == 0:
                final = []
            current_v = matrix[:,i]

            min = np.min(current_v)
            max = np.max(current_v)
            var = np.var(current_v)
            final = np.append(final, min)
            final = np.append(final, max)
            final = np.append(final, var)

        else:
            #extract and transpose the corresponding coloumn vector
            if gradient == 0:
                grad = matrix[:,i]
            elif gradient == 1:
                grad = np.gradient(matrix[:, i])
            elif gradient == 2:
                grad = np.gradient(matrix[:, i])
                grad = np.gradient(grad)


            new_vect = resize_vect(grad,row)
            if i == 0:
                final = new_vect
            else:
                final = np.append(final,new_vect)
    return final


def z_normalize(data):
    return StandardScaler().fit(data).transform(data)