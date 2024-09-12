import h5py

def are_matrices_equal(matrix1, matrix2):
    """
    This function takes in two matrices and returns True if they are equal, False otherwise.
    """
    h5f1 = h5py.File(matrix1, 'r')
    melamp1 = h5f1['mel'][:]

    h5f2 = h5py.File(matrix2, 'r')
    melamp2 = h5f2['feats'][:]

    matrix1 = melamp1
    matrix2 = melamp2

    # Check if the dimensions of the matrices are the same
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False

    # Check if each element in the matrices is the same
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] != matrix2[i][j]:
                print('not same')

    # If all elements are the same, return True
    print('same')

if __name__ == '__main__':
    matrix_1 = '/home/zhaorn/radarmic/Radio_audio/test/TIMIT_mel/norm_mel/audio/FAKS0/SI1573.h5'
    matrix_2 = '/home/zhaorn/wavegan/ParallelWaveGAN/egs/TIMIT/voc1/dump/train_nodev_all/norm/dump.1/FAKS0_SI1573.h5'
    are_matrices_equal(matrix_1, matrix_2)