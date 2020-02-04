"""i/o functions"""
import pickle


def save_single_source_results(results, filename='single_source_results'):
    """save single source results in a pickle"""
    pickle.dump(results, open(filename + '.pkl', 'wb'))


def load_single_source_results(filename='single_source_results'):
    """load single source results from a pickle"""
    return pickle.load(open(filename + '.pkl', 'rb'))
