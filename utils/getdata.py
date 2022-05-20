import numpy as np


def get_data():
    img = np.load('./MovingMnist/mnist_test_seq.npy').transpose(1, 0, 2, 3)

    train_data = img[:int(0.6 * len(img)), :, :, :]
    valid_data = img[int(0.6 * len(img)):int(0.8 * len(img)), :, :, :]
    test_data = img[int(0.8 * len(img)):, :, :, :]

    print('Train Shape: ', train_data.shape, '| Valid Shape: ', valid_data.shape, '| Test Shape: ', test_data.shape)
    return train_data, valid_data, test_data
