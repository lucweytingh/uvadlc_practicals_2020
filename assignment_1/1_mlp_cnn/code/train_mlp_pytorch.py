"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = "100"
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100


# Directory in which cifar data is saved
DATA_DIR_DEFAULT = "./cifar10/cifar-10-batches-py"

FLAGS = None
N_CLASSES = 10


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 1D int array of size batch_size. Ground truth labels for
              each sample in the batch.
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    # Convert to numpy.
    predictions = predictions.numpy()

    # Calculate accuracy.
    accuracy = (np.argmax(predictions, axis=1) == targets).sum() / len(targets)

    return accuracy


def train(mlp, data, optimizer, loss_module):
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    softmax = nn.Softmax(dim=1)
    losses = []
    accuracies = []
    epoch = -1
    step = 0

    X_test_imgs, Y_test = data["test"].images, data["test"].labels
    X_test = imgs_to_tensors(X_test_imgs)

    while step <= FLAGS.max_steps:
        # Retrieve the images.
        X_imgs, Y = data["train"].next_batch(FLAGS.batch_size)
        X = imgs_to_tensors(X_imgs)

        # Run the model on the input data.
        Y_pred = mlp(X)

        # Calculate the loss.
        loss = loss_module(Y_pred, torch.from_numpy(Y))

        optimizer.zero_grad()

        # Perform backpropagation.
        loss.backward()

        # Update weights.
        optimizer.step()

        if epoch != data["train"].epochs_completed:
            losses.append(loss.item())
            with torch.no_grad():
                accuracies.append(accuracy(softmax(mlp(X_test)), Y_test))
            epoch = data["train"].epochs_completed
            print("epoch: {0}, loss: {1:.3f}".format(epoch, loss.item()))
        step += 1

        with torch.no_grad():
            if not step % FLAGS.eval_freq:
                test_accuracy = accuracy(softmax(mlp(X_test)), Y_test)
                print(
                    "step: {0}, accuracy: {1:.3f}".format(step, test_accuracy)
                )
    plot_loss_acc(accuracies, losses)

    with torch.no_grad():
        final_test_accuracy = accuracy(softmax(mlp(X_test)), Y_test)
        print("final accuracy: {:.3f}".format(final_test_accuracy))


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + " : " + str(value))


def imgs_to_tensors(imgs):
    vectorized = imgs.reshape((imgs.shape[0], np.prod(imgs.shape[1:])))
    return torch.from_numpy(vectorized)


def plot_loss_acc(accuracies, loss):
    epochs = np.arange(len(accuracies))
    plt.plot(epochs, loss, label="Loss")
    plt.plot(epochs, accuracies, label="Accuracy")
    plt.xlabel("epochs")

    plt.legend()
    plt.show()


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [
            int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units
        ]
    else:
        dnn_hidden_units = []

    # neg_slope = FLAGS.neg_slope

    data = cifar10_utils.get_cifar10(
        FLAGS.data_dir, one_hot=False, validation_size=0
    )

    img_shape = data["train"].images[0].shape

    # print(np.prod(img_shape), dnn_hidden_units, N_CLASSES)
    mlp = MLP(np.prod(img_shape), dnn_hidden_units, N_CLASSES)
    print(mlp)

    optimizer = optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    loss_module = nn.CrossEntropyLoss()

    # run the training operation
    train(mlp, data, optimizer, loss_module)


# USED FOR TESTING
class FLAGS:
    data_dir = DATA_DIR_DEFAULT
    max_steps = 20000
    dnn_hidden_units = "50,40,50"
    learning_rate = LEARNING_RATE_DEFAULT * 1.5
    batch_size = 100
    eval_freq = 500


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dnn_hidden_units",
        type=str,
        default=DNN_HIDDEN_UNITS_DEFAULT,
        help="Comma separated list of number of units in each hidden layer",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=MAX_STEPS_DEFAULT,
        help="Number of steps to run trainer.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE_DEFAULT,
        help="Batch size to run trainer.",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=EVAL_FREQ_DEFAULT,
        help="Frequency of evaluation on the test set",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR_DEFAULT,
        help="Directory for storing input data",
    )
    FLAGS, unparsed = parser.parse_known_args()

    main()
