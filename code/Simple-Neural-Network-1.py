import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = np.array([[0], [1], [1], [0]])

    learning_rate = 0.5
    error = np.zeros((10000, 1))

    # Create 'connection weight'
    weight1 = np.random.random((3, 2))
    weight2 = np.random.random((3, 1))

    # Layer1 : 4x2 to 4x3
    x_data1 = np.zeros((x_data.shape[0], x_data.shape[1] + 1))
    x_data1[:, :-1] = x_data
    x_data1[:, -1] = [1, 1, 1, 1]

    # Iteration
    for k in range(10000):
        # Layer1 : Neural Net
        layer1_1 = np.matmul(x_data1, weight1)

        # Layer1 : Sigmoid
        layer1_2 = 1 / (1 + np.exp(-layer1_1))

        # Layer2 : 4x2 to 4x3
        x_data2 = np.zeros((layer1_2.shape[0], layer1_2.shape[1] + 1))
        x_data2[:, :-1] = layer1_2
        x_data2[:, -1] = [1, 1, 1, 1]

        # Layer2 : Neural Net
        layer2_1 = np.matmul(x_data2, weight2)

        # Layer2 : Sigmoid
        layer2_2 = 1 / (1 + np.exp(-layer2_1))

        # define 'variable'
        sum_v = np.zeros((4, 3))
        sum_w1n = np.zeros((4, 3))
        sum_w2n = np.zeros((4, 3))

        # En / Vkj
        for i in range(4):
            sum_v[i] = -(y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i] * np.reshape(x_data2[i], (1, 3))

        # En / W2i
        for i in range(4):
            sum_w2n[i] += np.reshape(-np.reshape(x_data1[i], (3, 1)) * layer1_2[i][1] * (1 - layer1_2[i][1]) * float(weight2[1] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # En / W1i
        for i in range(4):
            sum_w1n[i] += np.reshape(-np.reshape(x_data1[i], (3, 1)) * layer1_2[i][0] * (1 - layer1_2[i][0]) * float(weight2[0] * (y_data - layer2_2)[i] * layer2_2[i] * (1 - layer2_2)[i]), (-1))

        # Update the weights
        for i in range(3):
            weight2[i] -= learning_rate * (sum_v[0][i] + sum_v[1][i] + sum_v[2][i] + sum_v[3][i])

        for i in range(3):
            weight1[i][0] = weight1[i][0] - learning_rate * (sum_w1n[0][i] + sum_w1n[1][i] + sum_w1n[2][i] + sum_w1n[3][i])
            weight1[i][1] = weight1[i][1] - learning_rate * (sum_w2n[0][i] + sum_w2n[1][i] + sum_w2n[2][i] + sum_w2n[3][i])

        for i in range(4):
            error[k] += (y_data[i] - layer2_2[i]) ** 2
        error[k] /= 2

    plt.plot(error)
    plt.xticks([1, 876, 1751, 2626, 3501, 4376, 5251, 6126, 7001, 7876, 8751, 9626])
    plt.yticks([0, 0.5, 1, 1.5, 2, 2.5])
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title('Error graph')
    plt.show()

