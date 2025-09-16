import torch
from collections import Counter
import matplotlib.pyplot as plt


def error_rate(predicted, ground_truth):
    """
    Calculates the number of incorrect characters for each sequence in a batch
    """
    predicted_indexes = predicted.argmax(2)
    incorrect = (predicted_indexes != ground_truth)
    incorrect = incorrect.sum(dim=0)
    return incorrect.tolist()

def plot_errors_fn(errors):
    errors = Counter(errors)
    x = list(errors.keys())
    y = list(errors.values())
    plt.bar(x, y)
    plt.xlabel("Number of errors per string")
    plt.ylabel("Frequency")
    plt.show()


def loss(output, y, loss_fn):
    """
    Calculates the loss for a batch, reshaping the tensors appropriately.
    """
    output_dim = output.shape[-1]
    output = output[1:].contiguous().view(-1, output_dim)
    y = y[1:].contiguous().view(-1)
    return loss_fn(output, y)