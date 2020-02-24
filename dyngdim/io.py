"""i/o functions"""
import pickle


def save_local_dimensions(times, local_dimensions, filename="local_dimensions"):
    """save local dimensions in a pickle"""
    pickle.dump([times, local_dimensions], open(filename + ".pkl", "wb"))


def save_all_sources_relative_dimensions(
    relative_dimensions, filename="all_sources_relative_dimensions"
):
    """save all source relative dimensions in a pickle"""
    pickle.dump(relative_dimensions, open(filename + ".pkl", "wb"))


def load_local_dimensions(filename="local_dimensions"):
    """load local dimensionfrom a pickle"""
    return pickle.load(open(filename + ".pkl", "rb"))


def load_all_sources_relative_dimensions(filename="all_sources_relative_dimensions"):
    """load all source relative dimensionfrom a pickle"""
    return pickle.load(open(filename + ".pkl", "rb"))


def save_single_source_results(results, filename="single_source_results"):
    """save single source results in a pickle"""
    pickle.dump(results, open(filename + ".pkl", "wb"))


def load_single_source_results(filename="single_source_results"):
    """load single source results from a pickle"""
    return pickle.load(open(filename + ".pkl", "rb"))
