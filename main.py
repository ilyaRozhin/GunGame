import collections
import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from NNetwork import Net

G = 9.8
FIRST_HIDDEN_SIZE = 128
SECOND_HIDDEN_SIZE = 16
BATCH_SIZE = 16
PERCENTILE = 0.7
TRAINING_SAMPLES = 20000
TEST_SAMPLES = 300


def sample(alpha, s):
    return math.sqrt(s * G / math.sin(2*alpha))


def generate_data(min_alpha, max_alpha, min_distance, max_distance):
    alpha = random.uniform(min_alpha, max_alpha)
    s = random.uniform(min_distance, max_distance)
    return (alpha, s), sample(alpha, s)


if __name__ == "__main__":

    print(sample(1, 1000))

    """ Create Net """

    net = Net(FIRST_HIDDEN_SIZE, SECOND_HIDDEN_SIZE)

    """ Create Train, Test Data """

    min_a = math.radians(20)
    max_a = math.radians(50)
    min_s = 1
    max_s = 1000
    train_data = []
    train_labels = []

    for _ in range(TRAINING_SAMPLES * BATCH_SIZE):
        train_sample, train_label = generate_data(min_a, max_a, min_s, max_s)
        train_data.append(train_sample)
        train_labels.append(train_label)

    min_a_t = math.radians(51)
    max_a_t = math.radians(70)
    min_s_t = 1001
    max_s_t = 2000
    test_data = []
    test_labels = []

    for _ in range(TEST_SAMPLES * BATCH_SIZE):
        test_sample, test_label = generate_data(min_a_t, max_a_t, min_s_t, max_s_t)
        test_data.append(test_sample)
        test_labels.append(test_label)

    """ Network training"""

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=net.parameters(), lr=0.01)
    ids = 0

    train_data_info = []
    batch_samples = []
    batch_labels = []
    for i in range(len(train_data)):
        ids += 1
        batch_samples.append(train_data[i])
        batch_labels.append(train_labels[i])
        if ids % BATCH_SIZE == 0:
            samples_t = torch.FloatTensor(np.array(batch_samples))
            labels_t = torch.FloatTensor(batch_labels).reshape(shape=(BATCH_SIZE, 1))
            out = net(samples_t)
            loss = loss_function(out, labels_t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_data_info.append(loss.item())
            batch_samples = []
            batch_labels = []
    plt.title("Loss function graph")
    plt.xlabel("Number of iter")
    plt.ylabel("Loss value")
    plt.grid(visible=True)
    plt.plot(range(0, len(train_data_info)), train_data_info)
    plt.savefig("result_loss.png")
    plt.show()

    """ Network testing """

    ids = 0
    test_data_info = []
    batch_samples = []
    batch_labels = []
    for i in range(len(test_data)):
        ids += 1
        batch_samples.append(test_data[i])
        batch_labels.append(test_labels[i])
        if ids % BATCH_SIZE == 0:
            samples_t = torch.FloatTensor(np.array(batch_samples))
            labels_t = torch.FloatTensor(batch_labels).reshape(shape=(BATCH_SIZE, 1))
            out = net(samples_t)
            loss = loss_function(out, labels_t)
            delta = abs(out.detach().numpy() - labels_t.detach().numpy()) / labels_t.detach().numpy()
            test_data_info.append(delta.mean())
            batch_samples = []
            batch_labels = []

    max_value = max(test_data_info)
    min_value = min(test_data_info)
    max_index = test_data_info.index(max_value)
    min_index = test_data_info.index(min_value)

    plt.title("Result evaluation")
    plt.xlabel("Number of test")
    plt.ylabel("Mean delta value")
    plt.plot(range(0, len(test_data_info)), test_data_info)
    plt.plot([max_index, min_index], [max_value, min_value], "ro")
    plt.text(max_index + 3, max_value, "Верхняя граница: " + str(round(max_value, 3)))
    plt.text(min_index + 3, min_value, "Нижняя граница: " + str(round(min_value, 3)))
    plt.grid(visible=True)
    plt.savefig("result_ev.png")
    plt.show()

