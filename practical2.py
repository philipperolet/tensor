import torch
import dlc_practical_prologue as prologue


def nn_classification(training_inputs, training_labels, x):
    if (len(x.size()) == 2):
        x = x.view(x.size(0), 1, x.size(1))
    index = torch.argmin(torch.norm(training_inputs - x, dim=2), 1)
    return training_labels[index]


def test_exo1():
    train_inputs = torch.randint(0, 100, [10, 2], dtype=torch.float)
    train_labels = torch.randint(0, 100, [10], dtype=torch.float)
    print(train_inputs, train_labels,
          nn_classification(train_inputs, train_labels, torch.Tensor([[50.0, 50.0]])))
    print(train_inputs, train_labels,
          nn_classification(train_inputs, train_labels, torch.Tensor([[50.0, 50.0], [20.0, 20.0]])))


def nb_errors(train_data, train_labels, test_data, test_labels, mean, proj):
    proj_train_data = (train_data - mean) @ proj.t() if proj is not None else (train_data - mean)
    proj_test_data = (test_data - mean) @ proj.t() if proj is not None else (test_data - mean)
    return torch.sum(
        torch.sign(
            torch.abs(
                test_labels - nn_classification(proj_train_data, train_labels, proj_test_data))))


def pca(x):
    x_0 = x - x.mean(0)
    values, vectors = torch.eig(x_0.t() @ x_0, eigenvectors=True)
    return x.mean(0), torch.index_select(vectors.t(), 0, torch.sort(values[:, 0], descending=True)[1])


def test_pca():
    x = torch.tensor([[2.5, 2.4],
                      [0.5, 0.7],
                      [2.2, 2.9],
                      [1.9, 2.2],
                      [3.1, 3.0],
                      [2.3, 2.7],
                      [2, 1.6],
                      [1, 1.1],
                      [1.5, 1.6],
                      [1.1, 0.9]])
    eigen_vectors = torch.tensor([[-0.677873399, -0.735178656], [-0.735178656, 0.677873399]])
    assert torch.norm(eigen_vectors - pca(x)[1]) < 0.0001, "Error in PCA Computation."


test_pca()

train_input, train_target, test_input, test_target = prologue.load_data()

mnist_mean = train_input.mean(0)
mnist_sample_size = train_input.size(0)
mnist_dim = train_input.size(1)


def mnist_errors(mean, proj):
    return nb_errors(train_input, train_target, test_input, test_target, mean, proj)


rp100_errors = mnist_errors(mnist_mean, torch.empty([100, mnist_dim]).normal_())
rp200_errors = mnist_errors(mnist_mean, torch.empty([300, mnist_dim]).normal_())
np_errors = mnist_errors(mnist_mean, None)

print("Errors for 100d random subspace projection : {}".format(rp100_errors))
print("Errors for 200d random subspace projection : {}".format(rp200_errors))
print("Errors for full NN : {}".format(np_errors))


for i in [3, 50, 100, 200]:
    pca_errors = mnist_errors(mnist_mean, pca(train_input)[1][0:i])
    print("Errors for PCA with {} vectors: {}".format(i, pca_errors))
