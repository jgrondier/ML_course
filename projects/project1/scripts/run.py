from implementations import *
import proj1_helpers as helpers
import numpy as np




yb, raw_data, _ = helpers.load_csv_data("../data/train.csv", False)
_, raw_test_data, ids = helpers.load_csv_data("../data/test.csv", False)
lambdas = [1e-5, 2.5e-5, 1e-5, 1e-4]
degrees = [1,7,2,7]

preds = []
ids_final = []

for i in tqdm(range(len(degrees))):
    test_data = prepare_data(raw_test_data, analyse_data(raw_data), degrees[i])
    w = train_ridge_rmse(yb, raw_data, lambdas[i], degrees[i])[i]
    for idx, ev in tqdm(enumerate(test_data)):
        pri = int(raw_test_data[idx][22])
        if pri == i:
            x = w.dot(ev)
            preds.append(-1 if x < 0 else 1)
            ids_final.append(ids[idx])

helpers.create_csv_submission(ids_final, preds, "results.csv")
