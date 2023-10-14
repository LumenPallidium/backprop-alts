from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

def _prepare_for_epochs():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: x.view(-1))])

    mnist = datasets.MNIST('../data', train=True, download=True,
                            transform=transform)
    mnist_val = datasets.MNIST('../data', train=False, download=True,
                            transform=transform)

    accs = []
    errors = []

    return mnist, mnist_val, accs, errors

def mnist_test(net,
               batch_size = 256, 
               n_epochs = 3, 
               n_labels = 10, 
               save_every = 10,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    with torch.no_grad():

        mnist, mnist_val, accs, errors = _prepare_for_epochs()

        for epoch in range(n_epochs):
            train_loader = torch.utils.data.DataLoader(mnist, 
                                                        batch_size = batch_size, 
                                                        shuffle = True)
            val_loader = torch.utils.data.DataLoader(mnist_val,
                                                        batch_size = batch_size,
                                                        shuffle = True)
            epoch_accs = []

            for i, (x, y) in tqdm(enumerate(val_loader)):
                x = x.to(device)
                y = y.to(device)
                y_copy = y.clone().detach()
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                y_hat = net(x)
                y_out = torch.argmax(y_hat, dim = 1)
                acc = (y_out == y_copy).float().mean()
                epoch_accs.append(acc.item())

            print(f"Epoch {epoch} Accuracy: {np.mean(epoch_accs)}")

            accs.extend(epoch_accs)

            for i, (x, y) in tqdm(enumerate(train_loader)):
                x = x.to(device)
                y = y.to(device)
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                error = net.train_step(x, y)

                if i % save_every == 0:
                    errors.append(error[0])
        
        plt.plot(accs)
        plt.show()

        return accs, errors, y