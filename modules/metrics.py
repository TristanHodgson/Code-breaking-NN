import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def error_rate(predicted, ground_truth, PAD_TOKEN=27):
    """
    Calculates the number of incorrect characters for each sequence in a batch
    """
    predicted_indexes = predicted.argmax(2)
    # mask = (ground_truth != PAD_TOKEN)
    incorrect = (predicted_indexes != ground_truth)  # & mask
    incorrect = incorrect.sum(dim=0)
    return incorrect.tolist()


def error_stats(errors, acceptable=10):
    errors_np = np.array(errors)
    mean = np.mean(errors_np)
    sd = np.std(errors_np)
    maximum = np.max(errors_np)
    acceptable_errors_count = np.sum(errors_np < acceptable)
    acceptable_errors_prop = acceptable_errors_count/len(errors)
    return mean, sd, maximum, acceptable_errors_prop


def plot_errors(errors, clip=100, acceptable=10):
    mean, sd, maximum, acceptable_errors_prop = error_stats(errors, acceptable)

    errors = Counter(errors)

    plot_counts = {}
    cliped_count = 0
    for value, count in errors.items():
        if value > clip:
            cliped_count += count
        else:
            plot_counts[value] = count
    if cliped_count > 0:
        plot_counts[clip+1] = cliped_count

    x = sorted(plot_counts.keys())
    y = [plot_counts[key] for key in x]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.bar(x, y)
    plt.xlabel("Number of errors per string")
    plt.ylabel("Frequency")
    plt.title(f"Mean: {mean:.1f} | Sd: {sd:.1f} | Proportion with acceptable errors: {acceptable_errors_prop:.2%} | Max errors: {maximum}\nClipped at {clip}, Acceptable means < {acceptable} character errors per string")
    plt.show()


def loss(output, y, loss_fn):
    """
    Calculates the loss for a batch, reshaping the tensors appropriately.
    """
    output_dim = output.shape[-1]
    output = output[1:].contiguous().view(-1, output_dim)
    y = y[1:].contiguous().view(-1)
    return loss_fn(output, y)
