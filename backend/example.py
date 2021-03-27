import prediction, zone_splitting, path_planning


DIR_PATH = '/content/Impact_challenge'

predictor = prediction.MakePrediction(DIR_PATH)
pred = predictor.prediction()

zones_split = zone_splitting.kmeans_subdivision(pred, DIR_PATH, 2)

paths = path_planning.path_planning(zones_split, DIR_PATH)
