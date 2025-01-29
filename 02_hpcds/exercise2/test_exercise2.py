from tkinter.tix import INTEGER

import numpy as np
import pytest
import exercise2

matr_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_4 = np.array([[31, 38, 45], [70, 86, 102], [109, 134, 159]])

def generate_test(n, testsize):
    ret = []
    np.random.seed(seed=1234)
    for i in range(testsize):
        a = np.random.randint(low=0, high=2**20-1, size=(n,n))
        b = np.random.randint(low=0, high=2**20-1, size=(n,n))
        c = np.random.randint(low=0, high=2**20-1, size=(n,n))
        expected = c + a@b
        ret.append((a, b, c, expected))
    return ret

@pytest.mark.parametrize("a, b, c, expected", generate_test(5, 5))
def test_mult_add_py(a, b, c, expected):
    assert np.array_equal(exercise2.mult_add_py(a, b, c), expected)
