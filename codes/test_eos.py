import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np   
import os          
from torch.nn.utils import parameters_to_vector, vector_to_parameters         
from scipy.sparse.linalg import LinearOperator, eigsh       
import time      
from torch.utils.data import TensorDataset

def iterate_dataset(dataset, batch_size: int):
    """Iterate through a dataset, yielding batches of data."""
    loader =  torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for (batch_X, batch_y) in loader:
        yield batch_X.cuda(), batch_y.cuda()




def compute_hvp(network: nn.Module, loss_fn: nn.Module,
                dataset, vector, physical_batch_size: int = 1000):
    """Compute a Hessian-vector product."""
    start = time.time()
    p = len(parameters_to_vector(network.parameters()))
    n = len(dataset)
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for (X, y) in iterate_dataset(dataset, physical_batch_size):
        loss = loss_fn(network(X), y) / n
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    print(f"compute_hvp took {time.time() - start} seconds")
    return hvp


def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return np.ascontiguousarray(evals[::-1]).copy().astype(np.float32), \
           np.ascontiguousarray(np.flip(evecs, -1)).copy().astype(np.float32)


def get_hessian_eigenvalues(network: nn.Module, loss_fn: nn.Module, dataset,
                            neigs=6, physical_batch_size=1000):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, loss_fn, dataset,
                                          delta, physical_batch_size=physical_batch_size).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


# Build the dataset
# TODO: add data augmentation
train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(32, 4),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])


test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
#                                         download=True, transform=train_transform)
# only take random 5000 samples from the training set
#trainset, _ = torch.utils.data.random_split(trainset, [1000, 49000])

def center(X_train: np.ndarray, X_test: np.ndarray):
    mean = X_train.mean(0)
    return X_train - mean, X_test - mean

def standardize(X_train: np.ndarray, X_test: np.ndarray):
    std = X_train.std(0)
    return (X_train / std, X_test / std)

def flatten(arr: np.ndarray):
    return arr.reshape(arr.shape[0], -1)

def unflatten(arr: np.ndarray, shape):
    return arr.reshape(arr.shape[0], *shape)

def _one_hot(tensor, num_classes: int, default=0):
    M = F.one_hot(tensor, num_classes)
    M[M == 0] = default
    return M.float()

def make_labels(y, loss):
    if loss == "ce":
        return y
    elif loss == "mse":
        return _one_hot(y, 10, 0)

DATASETS_FOLDER = '../data'
def load_cifar(loss: str) -> (TensorDataset, TensorDataset):
    cifar10_train = torchvision.datasets.CIFAR10(root=DATASETS_FOLDER, download=True, train=True)
    cifar10_test = torchvision.datasets.CIFAR10(root=DATASETS_FOLDER, download=True, train=False)
    X_train, X_test = flatten(cifar10_train.data / 255), flatten(cifar10_test.data / 255)
    y_train, y_test = make_labels(torch.tensor(cifar10_train.targets), loss), \
        make_labels(torch.tensor(cifar10_test.targets), loss)
    center_X_train, center_X_test = center(X_train, X_test)
    standardized_X_train, standardized_X_test = standardize(center_X_train, center_X_test)
    train = TensorDataset(torch.from_numpy(unflatten(standardized_X_train, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_train)
    test = TensorDataset(torch.from_numpy(unflatten(standardized_X_test, (32, 32, 3)).transpose((0, 3, 1, 2))).float(), y_test)
    return train, test


trainset, testset = load_cifar("ce")
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5000,
                                            shuffle=True, num_workers=32)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=test_transform)
# split test set into validation set and test set
testset, valset = torch.utils.data.random_split(testset, [8000, 2000])
test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)
val_loader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=True, num_workers=2)


# build simple MLP
class MLPNet(nn.Module):
    def __init__(self, layers=1, hidden_units=200):
        super().__init__()
        self.layers = layers
        for i in range(layers):
            if i == 0:
                self.fc1 = nn.Linear(32 * 32 * 3, hidden_units)
            else:
                setattr(self, 'fc{}'.format(i+1), nn.Linear(hidden_units, hidden_units)) # TODO: experiment with different number of hidden units
        self.classifier = nn.Linear(hidden_units, 10)
        
    
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        for i in range(self.layers):
            x = getattr(self, 'fc{}'.format(i+1))(x)
            x = F.relu(x)
        x = self.classifier(x)
        return x
    

def get_activation(activation: str):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'hardtanh':
        return torch.nn.Hardtanh()
    elif activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == "tanh":
        return torch.nn.Tanh()
    elif activation == "softplus":
        return torch.nn.Softplus()
    elif activation == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise NotImplementedError("unknown activation function: {}".format(activation))
    
def fully_connected_net(widths = [200, 200],  activation='tanh', bias = True) -> nn.Module:
    modules = [nn.Flatten()]
    for l in range(len(widths)):
        prev_width = widths[l - 1] if l > 0 else 32 * 32 * 3
        modules.extend([
            nn.Linear(prev_width, widths[l], bias=bias),
            get_activation(activation),
        ])
    modules.append(nn.Linear(widths[-1], 10, bias=bias))
    return nn.Sequential(*modules)
    

def compute_loss_on_whole_dataset(net, dataloader, loss_fn):
    # compute loss on whole dataset
    total_loss = 0
    total_outputs = []
    total_labels = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()
        total_outputs.append(outputs)
        total_labels.append(labels)
    return total_loss / len(dataloader), torch.cat(total_outputs, dim=0), torch.cat(total_labels, dim=0)


#  ========== [Classification task with MLP] ================
# build optimizer and loss function


def take_first(dataset: TensorDataset, num_to_keep: int):
    return TensorDataset(dataset.tensors[0][0:num_to_keep], dataset.tensors[1][0:num_to_keep])


experiment_name = 'MLP_1_layer_64_hidden_units_cls'
experiment_path = './experiments/{}'.format(experiment_name)
os.makedirs(experiment_path, exist_ok=True)
loss_fn = nn.CrossEntropyLoss()
# one layer MLP
net = fully_connected_net().cuda()
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
max_epochs = 5000
# train the network and plot the loss curve
train_losses = []
# Grdieint descent over the whole training set
eig_freq = 50
train_loss, train_acc, test_loss, test_acc = torch.zeros(max_epochs), torch.zeros(max_epochs), torch.zeros(max_epochs), torch.zeros(max_epochs)
eigenvalues_list = []
for epoch in range(max_epochs):
    abridged_train = take_first(trainset, 1000)
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = net(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    
    train_loss[epoch], train_outputs, train_labels = compute_loss_on_whole_dataset(net, train_loader, loss_fn)
    train_acc[epoch] = (train_outputs.argmax(dim=1) == train_labels).float().mean()
    test_loss[epoch], test_outputs, test_labels = compute_loss_on_whole_dataset(net, test_loader, loss_fn)
    test_acc[epoch] = (test_outputs.argmax(dim=1) == test_labels).float().mean()
    # log the loss
    print('Epoch {}: train loss {}, train acc {}, test loss {}, test acc {}'.format(epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch], test_acc[epoch]))
    
    # get the eigenvalues of the Hessian
    if epoch % eig_freq == 0:
        eigenvalues = get_hessian_eigenvalues(net, loss_fn, abridged_train, 2, 1000)
        print('Epoch {}: eigenvalues {}'.format(epoch, eigenvalues))
        eigenvalues_list.append(eigenvalues)
    # update the weights
   
    torch.save(net.state_dict(), os.path.join(experiment_path, 'epoch_{}.pth'.format(epoch)))
    
# save and plot the loss & acc curve
np.save(os.path.join(experiment_path, 'train_loss.npy'), train_loss.detach().cpu().numpy())
np.save(os.path.join(experiment_path, 'train_acc.npy'), train_acc.detach().cpu().numpy())
np.save(os.path.join(experiment_path, 'test_loss.npy'), test_loss.detach().cpu().numpy())
np.save(os.path.join(experiment_path, 'test_acc.npy'), test_acc.detach().cpu().numpy())
np.save(os.path.join(experiment_path, 'eigenvalues.npy'), np.stack(eigenvalues_list, axis=0))
plt.figure()
plt.plot(np.arange(max_epochs), train_loss.detach().cpu().numpy(), label='train loss')
plt.plot(np.arange(max_epochs), test_loss.detach().cpu().numpy(), label='test loss')
plt.legend()
plt.savefig(os.path.join(experiment_path, 'loss_curve.png'))
plt.figure()
plt.plot(np.arange(max_epochs), train_acc.detach().cpu().numpy(), label='train acc')
plt.plot(np.arange(max_epochs), test_acc.detach().cpu().numpy(), label='test acc')
plt.legend()
plt.savefig(os.path.join(experiment_path, 'acc_curve.png'))
# plot the eigenvalues 
# the largest eigenvalue is the sharpness of the loss function using scatter plot
plt.figure()
plt.scatter(np.arange(max_epochs // eig_freq), np.stack(eigenvalues_list, axis=0)[:, 0])
plt.axhline(y= 2 /lr, color='r', linestyle='dotted')
plt.savefig(os.path.join(experiment_path, 'eigenvalues.png'))
# the smallest eigenvalue is the curvature of the loss function using scatter plot








    
