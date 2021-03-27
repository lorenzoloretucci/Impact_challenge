import backend.prediction
import backend.zone_splitting
import backend.path_planning


def get_paths(n_trucks):
    DIR_PATH = '.'

    predictor = backend.prediction.MakePrediction(DIR_PATH)
    pred = predictor.prediction()

    zones_split = backend.zone_splitting.kmeans_subdivision(pred, DIR_PATH, n_trucks)

    return backend.path_planning.path_planning(zones_split, DIR_PATH)
