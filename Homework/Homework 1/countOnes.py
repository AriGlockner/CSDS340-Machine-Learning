import numpy as np


def countOnesLoop(array):
    """
    Using a for loop to count the number of 1s in an array

    :param array: array to count 1s in
    :return: number of 1s in the array
    """

    count = 0
    for i in range(len(array)):
        if array[i] == 1:
            count += 1
    return count


def countOnesWhere(array):
    """
    Using np.where() to count the number of 1s in an array

    :param array: array to count 1s in
    :return: number of 1s in the array
    """
    # np.where(1, array).size
    return np.where(array[array == 1], 1, 0).size


# Create random test cases and print the results out
for j in range(10):
    a = np.random.randint(0, 3, 10)
    print(a, '\nloop:', countOnesLoop(a), '\nwhere:', countOnesWhere(a), '\n')
