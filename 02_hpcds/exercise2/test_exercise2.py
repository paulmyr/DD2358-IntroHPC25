from tkinter.tix import INTEGER

import numpy as np
import pytest
import exercise2

matr_01 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_02 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_03 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matr_04 = np.array([[31, 38, 45], [70, 86, 102], [109, 134, 159]])

matr_11 = np.identity(3)
matr_12 = np.identity(3)
matr_13 = np.zeros((3,3))
matr_14 = np.identity(3)

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
    res = exercise2.mult_add_py(a, b, c)
    assert np.array_equal(res, expected)


@pytest.mark.parametrize("a, b, c, expected", [
        (matr_01, matr_02, matr_03, matr_04),
        (matr_11, matr_12, matr_13, matr_14)],
        ids=["pos_ints", "zero_and_identity"]
)
def test_mult_add_py_manual(a, b, c, expected):
    assert np.array_equal(exercise2.mult_add_py(a, b, c), expected)
