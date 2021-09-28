"""
.. _example-ort-training-simple:

Onnxruntime Training with numpy data
====================================

Example inspired from `PyTorch: Defining new autograd functions
<https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#pytorch-defining-new-autograd-functions>`_.


.. contents::
    :local:

A simple example
++++++++++++++++

"""

import time
import math
from onnxruntime.capi.ort_trainer import (
    ORTTrainer, IODescription, ModelDescription)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Variable


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


def get_model(X, y, hidden_size=10):
    input_size = X_train.shape[1]
    output_size = 1 if len(y.shape) == 1 else y.shape[2]
    return NeuralNet(input_size, hidden_size, num_classes), F.MSELoss()


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
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.01, momentum=0.5)
        else:
            raise NotImplementedError("Unexpected optimizer %r." % optimizer)

    def _update_parameters(self, param_values):
        replaces = {'learning_rate': 'lr',
                    'Learning_Rate': 'lr'}
        for gr in self.optimizer.param_groups:
            for k, v in param_values.items():
                if k in replaces:
                    pin, pout = k, replaces[k]
                else:
                    pin, pout = k, k
                gr[pout] = param_values[pin][0]

    def train_step(self, data, target, *params):
        param_values = {k: v for k, v in zip(self.param_names, params)}
        self._update_parameters(param_values)

        y = self.model(data)
        loss = self.loss(y, target)
        loss.backward()
        self.optimizer.step()
        return loss, None

    def eval_step(self, data, fetches=None):
        "fetches is ignored in this case"
        out = self.model(data)
        return out

    def state_dict(self):
        return self.optimizer.state_dict()

    def save_as_onnx(self, filename, target_opset=14, batch_size=1):
        size = (batch_size, ) + tuple(self.model_desc.inputs_[0].shape_[1:])
        x = torch.randn(size, requires_grad=True).to(self.device)
        torch.onnx.export(
            self.model, x, filename, export_params=True,
            do_constant_folding=True,
            input_names=['input'], output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}})


def train_with_trainer(learning_rate, trainer, device,
                       X_train, y_train, epoch):
    actual_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)
        loss, _ = trainer.train_step(
            data, target, torch.tensor(
                [learning_rate]))
        args_log_interval = 100
        if batch_idx % args_log_interval == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '
                'lr={:.1g}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    learning_rate))
            items = loss.cpu().detach().numpy().item()
            actual_losses.append(items)

    return actual_losses


def test_with_trainer(trainer, device, X_train, y_train):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        data = data.reshape(data.shape[0], -1)
        output = F.log_softmax(
            trainer.eval_step(data, fetches=['probability']), dim=1)
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


class MNISTWrapper:

    def model_description(self):
        input_desc = IODescription('input1', ['batch', 784], torch.float32)
        label_desc = IODescription(
            'label', ['batch', ], torch.int64, num_classes=10)
        loss_desc = IODescription('loss', [], torch.float32)
        probability_desc = IODescription(
            'probability', ['batch', 10], torch.float32)
        return ModelDescription([input_desc, label_desc],
                                [loss_desc, probability_desc])


def test_training_testing(X_train, X_test, y_train, y_test,
                          trainer="ORT", device="cpu", epochs=1000,
                          learning_rate=0.001, save_model_epoch=100):
    torch.manual_seed(1)
    device_obj = torch.device(device)

    X_train = Variable(torch.from_numpy(X_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).float()
    X_test = Variable(torch.from_numpy(X_test)).float()
    y_test = Variable(torch.from_numpy(y_test)).float()

    print('-----------------------------')
    print("device=%r trainer=%r" % (device_obj, trainer))
    print('epochs=%r learning_rate=%r' % (epochs, learning_rate))
    print('-----------------------------')
    begin = time.perf_counter()

    mnist = MNISTWrapper()
    train_loader, test_loader = mnist.get_loaders()
    model, model_desc = mnist.get_model()
    model.to(device)
    for inp in model_desc.inputs_:
        print("   input: name=%r dty)e=%r shape=%r" % (
            inp.name_, inp.dtype_, inp.shape_))
    for inp in model_desc.outputs_:
        print("  output: name=%r dtype=%r shape=%r" % (
            inp.name_, inp.dtype_, inp.shape_))

    if trainer == "ORT":
        trainer_obj = get_ort_trainer(model, model_desc, device_obj)
    else:
        trainer_obj = get_torch_trainer(model, model_desc, device_obj)

    actual_losses = []
    actual_test_losses, actual_accuracies = [], []
    for epoch in range(1, epochs + 1):
        res = train_with_trainer(
            learning_rate, trainer_obj, device_obj,
            train_loader, epoch)
        actual_losses.extend(res)
        test_loss, accuracy = test_with_trainer(
            trainer_obj, device_obj, test_loader)
        actual_test_losses.append(test_loss)
        actual_accuracies.append(accuracy)

        if save_model_epoch > 0 and epoch % save_model_epoch == 0:
            name = "mninst_%s_%s_i%d.pt" % (trainer, device, epoch)
            state = {'rng_state': torch.get_rng_state(),
                     'model': trainer_obj.state_dict()}
            # with open(name, "wb") as f:
            # torch.save(state, f)
            torch.save(state, name)
            trainer_obj.save_as_onnx(name + ".onnx")

    print("actual_losses=", actual_losses)
    print("actual_test_losses=", actual_test_losses)
    print("actual_accuracies=", actual_accuracies)
    print("time=%r" % (time.perf_counter() - begin))


device = "cuda:0" if torch.cuda.is_available() else "cpu"
test_training_testing(
    trainer="torch", device=device, save_model_epoch=2)
test_training_testing(
    trainer="ORT", device=device, save_model_epoch=2)
