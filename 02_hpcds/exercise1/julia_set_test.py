import pytest
from julia_set_original import calc_pure_python

def test_julia_set_1k_squre_300_iter():
    """
    Assertion to test that JuliaSet code with 1000x1000 grid size
    and 300 max iterations returns the output with the expeected sum
    """
    expected_output_sum = 33219980
    actual_output_sum = sum(calc_pure_python(desired_width=1000, max_iterations=300))
    assert actual_output_sum == expected_output_sum

@pytest.mark.parametrize("size,max_iters,expected_output_sum", [(1000, 200, 23186920), (2000, 400, 173085144), (5000, 100, 327871908)])
def test_julia_set_parameterized(size, max_iters, expected_output_sum):
    """
    Parameterized tests for the julia-set code, illustrating how different grid-sizes and number
    of iterations can be used to create tests with a SINGLE function without having to 
    explicitly write tests for each case separately. 
    """
    actual_output_sum = sum(calc_pure_python(desired_width=size, max_iterations=max_iters))
    assert actual_output_sum == expected_output_sum
