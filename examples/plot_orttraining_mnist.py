"""
.. _example-ort-training:

Onnxruntime Training and MNIST
==============================


.. contents::
    :local:

A simple example
++++++++++++++++

"""

from numpy.testing import assert_allclose
from onnxruntime.capi.ort_trainer import (
    ORTTrainer, IODescription, ModelDescription)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def my_loss(x, target):
    return F.nll_loss(F.log_softmax(x, dim=1), target)


def get_ort_trainer(model, model_desc, device):
    return ORTTrainer(
        model, my_loss, model_desc, "SGDOptimizer", None,
        IODescription('Learning_Rate', [1, ], torch.float32),
        device, _opset_version=12)


class TorchTrainer:
    
    def __init__(self, model, loss, model_desc, optimizer, something,
                 param_names, device):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.model_desc = model_desc
        self.param_names = param_names
        self.device = device
        if optimizer == 'SGDOptimizer':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.0)
        else:
            raise NotImplementedError("Unexpected optimizer %r." % optimizer)

    def train_step(self, data, target, *params):
        param_value = {k: v for k, v in zip(self.param_names, params)}
        
        y = self.model(data)
        loss = self.loss(y, target)
        loss.backward()
        self.optimizer.step()
        return loss, None


def get_torch_trainer(model, model_desc, device):
    return TorchTrainer(
        model, my_loss, model_desc, "SGDOptimizer", None,
        ['learning_rate'], device=device)


class MNISTWrapper():

    def train_with_trainer(self, learningRate, trainer, device, train_loader,
                           epoch):
        actual_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = data.reshape(data.shape[0], -1)

            loss, _ = trainer.train_step(data, target, torch.tensor([learningRate]))

            args_log_interval = 100
            if batch_idx % args_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                items = loss.cpu().numpy().item()
                actual_losses.append(items)

        return actual_losses

    def test_with_trainer(self, trainer, device, test_loader):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = data.reshape(data.shape[0], -1)
                output = F.log_softmax(
                    trainer.eval_step(
                        (data),
                        fetches=['probability']),
                    dim=1)
                # sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, '
              'Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        return test_loss, correct / len(test_loader.dataset)

    def mnist_model_description():
        input_desc = IODescription('input1', ['batch', 784], torch.float32)
        label_desc = IODescription(
            'label', ['batch', ], torch.int64, num_classes=10)
        loss_desc = IODescription('loss', [], torch.float32)
        probability_desc = IODescription(
            'probability', ['batch', 10], torch.float32)
        return ModelDescription([input_desc, label_desc], [
                                loss_desc, probability_desc])

    def get_loaders(self):
        args_batch_size = 64
        args_test_batch_size = 1000

        kwargs = {'num_workers': 0, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=args_test_batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader

    def get_model(self):
        input_size = 784
        hidden_size = 500
        num_classes = 10

        # warning: changes the pytorch random generator state
        model = NeuralNet(input_size, hidden_size, num_classes)
        model_desc = MNISTWrapper.mnist_model_description()
        return model, model_desc


def testMNISTTrainingAndTesting(trainer="ORT", device="cpu"):
    torch.manual_seed(1)
    device = torch.device(device)
    print('-----------------------------')
    print("device=%r" % device)
    print('-----------------------------')

    mnist = MNISTWrapper()
    train_loader, test_loader = mnist.get_loaders()
    model, model_desc = mnist.get_model()
    
    if trainer == "ORT":
        trainer = get_ort_trainer(model, model_desc, device)
    else:
        trainer = get_torch_trainer(model, model_desc, device)

    learningRate = 0.01
    args_epochs = 2
    expected_losses = [
        2.333008289337158,
        1.0680292844772339,
        0.6300537586212158,
        0.5279903411865234,
        0.3710068166255951,
        0.4044453501701355,
        0.30482712388038635,
        0.4595026969909668,
        0.42305776476860046,
        0.4797358512878418,
        0.23006735742092133,
        0.48427966237068176,
        0.30716797709465027,
        0.3238796889781952,
        0.19543828070163727,
        0.3561663031578064,
        0.3089643716812134,
        0.37738722562789917,
        0.24883587658405304,
        0.30744990706443787]
    expected_test_losses = [0.31038025817871095, 0.25183824462890625]
    expected_test_accuracies = [0.9125, 0.9304]

    actual_losses = []
    actual_test_losses, actual_accuracies = [], []
    for epoch in range(1, args_epochs + 1):
        res = mnist.train_with_trainer(
                learningRate, trainer, device,
                train_loader, epoch)
        actual_losses.extend(res)

        test_loss, accuracy = mnist.test_with_trainer(
            trainer, device, test_loader)
        actual_test_losses = [*actual_test_losses, test_loss]
        actual_accuracies = [*actual_accuracies, accuracy]

        # if you update outcomes, also do so for resume from checkpoint test
        # args_checkpoint_epoch = 1
        # if epoch == args_checkpoint_epoch:
        # state = {'rng_state': torch.get_rng_state(),
        #          'model': trainer.state_dict()}
        # torch.save(state, get_name("ckpt_mnist.pt"))

    print("actual_losses=", actual_losses)
    print("actual_test_losses=", actual_test_losses)
    print("actual_accuracies=", actual_accuracies)

    # to update expected outcomes, enable pdb and run the test with -s
    # and copy paste outputs
    # import pdb; pdb.set_trace()
    rtol = 1e-01
    assert_allclose(
        expected_losses, actual_losses,
        rtol=rtol, err_msg="loss mismatch")
    assert_allclose(
        expected_test_losses, actual_test_losses,
        rtol=rtol, err_msg="test loss mismatch")
    assert_allclose(
        expected_test_accuracies, actual_accuracies,
        rtol=rtol, err_msg="test accuracy mismatch")


testMNISTTrainingAndTesting(trainer="torch")
