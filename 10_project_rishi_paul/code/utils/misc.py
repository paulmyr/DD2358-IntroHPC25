def continue_experiment(terminate_using, curr_time, curr_iters, end_count):
    if terminate_using == "I":
        return curr_iters < end_count
    else:
        return curr_time < end_count