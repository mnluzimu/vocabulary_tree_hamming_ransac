import numpy as np


def arr2bit(arr, n):
    result = 0
    for i in range(n):
        if arr[i] == 1:
            result = result | (1 << i)
    return result


def bit2arr(bitmap, n):
    arr = np.zeros(n).astype(np.int8)
    for i in range(n):
        if bitmap & (1 << i) != 0:
            arr[i] = 1
    return arr


if __name__ == "__main__":
    arr = np.array([1, 0, 0, 0])
    print(arr)
    result = arr2bit(arr, len(arr))
    print(bin(result))
    array = bit2arr(result, len(arr))
    print(array)
