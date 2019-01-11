import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

from resnet import ResNet

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

class_names = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = ResNet(3)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)

training_loss, validation_loss, validation_acc = [], [], []

def train(model, num_epoch=400):

    ite = 0
    for e in range(num_epoch):
        ########## Learning Rate Schedule
        if ite > 32000 and ite < 48000: set_lr(optimizer, 0.01)
        elif ite > 48000: set_lr(optimizer, 0.001)

        running_loss = 0.0
        for images, labels in trainloader:
            ite = ite + 1
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            training_loss.append(running_loss / len(trainloader))
            running_loss = 0.0
            val_acc = 0.0
            model.eval()

            with torch.no_grad():
                for images, labels in testloader:
                    images = images.to(device)
                    labels = labels.to(device)

                    out = model(images)
                    _, predicted = torch.max(out.data, 1)
                    equals = predicted == labels
                    val_acc += torch.mean(equals.type(torch.FloatTensor)).item()

                    loss = criterion(out, labels)
                    running_loss += loss.item()

            validation_acc.append(val_acc / len(testloader))
            validation_loss.append(running_loss / len(testloader))

            print(f'Epoch {e} / {num_epoch}: Training_loss: {training_loss[-1]:.3f}, Validation_loss: {validation_loss[-1]:.3f}, Validation_accuracy: {validation_acc[-1]:.3f} ')

            if validation_acc[-1] > np.max(validation_acc):
                model.save({
                    'epoch': e,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': validation_acc[-1]
                }, 'checkpoint.pth')


            model.train()

    print('Training Completed\n')
    model.save({
        'epoch': e,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': validation_acc[-1]
    }, 'final_checkpoint.pth')
    return model


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr


model = train(model, num_epoch=400)



