import numpy as np 
import torch

def l2_regularized_logistic_regression_loss(w, X = None, y=None):
    loss = 0
    n = len(y)

    for i in range(n):
        training_example, label = X[:, i], y[i]
        loss = loss + (torch.log(1 + torch.exp(-label * w.T @ training_example)))
    return float(loss)/float(n)


def read_a9a_dataset(file_path):
    dataset = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            first_blank_index = line.find(' ')
            label = int(line[:first_blank_index])

            training_feature_labels = {}
            feature = ""
            for idx in range(first_blank_index + 1, len(line)):
                if line[idx] != " ":
                    feature = feature + line[idx]
                else:
                    colon_index = feature.find(":")
                    training_feature_labels[int(feature[:colon_index])] = int(feature[colon_index + 1:])
                    feature = ""
            
            training_vector = np.zeros(124)
            for feature_label in training_feature_labels:
                training_vector[feature_label - 1] = 1
            training_vector[-1] = 1
            dataset.append(training_vector)
            labels.append(label)

    dataset = np.stack(dataset, axis=0).T
    labels = np.array(labels)

    assert dataset.shape == (124, 32561)
    assert labels.shape == (32561,)

    return torch.from_numpy(dataset).to(torch.float32), torch.from_numpy(labels).to(torch.float32)
