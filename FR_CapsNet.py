import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as utils
from collections import OrderedDict
import random
import numpy as np


class T_Caps(nn.Module):
    """
    Transitional capsule(T_Caps) layer, which is employed to transform features into capsules
    """
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=0, Capsule_size=9):
        super(T_Caps, self).__init__()
        # output_channels should be a multiple of Capsule_size
        self.Capsule_size = Capsule_size
        self.num_caps = int(output_channels / Capsule_size)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=self.num_caps, kernel_size=kernel_size,
                               stride=stride, padding=padding)

    def forward(self, x):
        capsules = self.conv(x)
        b, c, h, w = capsules.shape
        capsules = capsules.reshape(b, self.num_caps, self.Capsule_size, h, w)
        capsules = capsules.permute(0, 1, 3, 4, 2)
        probabilities = self.sigmoid(self.conv1(x))
        return capsules, probabilities


class Capslayer(nn.Module):
    """
    Capslayer is a capsule layer with fast routing. The low-level capsules and the high-level capsules are
    fully connected
    """
    def __init__(self, input_channels, output_channels, kernel_size, Capsule_size_in=9, Capsule_size_out=9):
        super(Capslayer, self).__init__()
        self.output_channels = output_channels
        self.k = kernel_size
        self.Transform = nn.Parameter(
            torch.randn(1, input_channels, output_channels, self.k, self.k, Capsule_size_out, Capsule_size_in))

    def transforms(self, x, T):
        # x.shape=batch_size, num_caps along the Channels, h, w, Capsule_size
        b, c, h, w, s = x.shape
        x = x.reshape(b, c, 1, h, w, s, 1).repeat(1, 1, self.output_channels, 1, 1, 1, 1)
        T = T.repeat(b, 1, 1, 1, 1, 1, 1)
        output = torch.matmul(T, x)
        return output.squeeze()

    def routing(self, x, p, epoch):
        T = self.Transform * 0.1
        predicted = self.transforms(x, T)
        # print(predicted.shape)
        b, in_c, out_c, h, w, s = predicted.shape
        # predicted.shape=batch_size, input_channels, output_channels, h, w, Capsule_size_out. at the last layer, h=w=k
        # p.shape=batch_size, input_channels, h, w.
        predicted = predicted.permute(0, 1, 3, 4, 2, 5).reshape(b, in_c * h * w, out_c, s)
        p = p.reshape(b, in_c * h * w, 1, 1)
        capsules_updated = (predicted * p).sum(dim=1) / ((p.sum(dim=1).reshape(b, 1, 1)) + 1e-8)
        diff = predicted - capsules_updated.reshape(b, 1, out_c, s)
        varience = (diff ** 2 * p).sum(dim=1) / ((p.sum(dim=1).reshape(b, 1, 1)) + 1e-8)
        p_updated = 1-varience.sum(dim=2) / ((torch.max(varience.sum(dim=2),dim=1)[0]).reshape(b,1)+1e-8)
        #p_updated = self.prob_updated(epoch, varience)
        return capsules_updated, p_updated

    def forward(self, x, p, epoch):
        return (self.routing(x, p, epoch))

    def prob_updated(self, epoch, varience):
        if epoch < 10:
            p_updated = torch.exp(-varience.sum(dim=2) * (0.1 * (epoch + 1)))
        else:
            p_updated = torch.exp(-varience.sum(dim=2))
        return p_updated


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size, num, total_num):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, int(growth_rate * (4 / total_num * num)),
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    def forward(self, x):
        # print(x.size())
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, int_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        num_in_channels = int_channels
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(num_in_channels,
                                        growth_rate, bn_size, i + 1, num_layers))
            num_in_channels += int(growth_rate * (4 / num_layers * (i + 1)))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=3,
                                          stride=1, padding=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseCaps_net(nn.Module):
    def __init__(self, growth_rate=32, block_config=(3, 3, 3),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseCaps_net, self).__init__()
        self.set_seed(1234)
        num_init_feature = 2 * growth_rate

        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):

            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            for j in range(num_layers):
                num_feature += int(growth_rate * (4 / num_layers * (j + 1)))
                # the \beta in the paper is set as 4
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        self.T_caps = T_Caps(input_channels=462, output_channels=32 * 9, kernel_size=3, stride=2, padding=0,
                             Capsule_size=9)
        self.Capslayer = Capslayer(input_channels=32, output_channels=10, kernel_size=3, Capsule_size_in=9,
                                   Capsule_size_out=9)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch):
        features = self.features(x)
        out, p = self.T_caps(features)
        out, p_1 = self.Capslayer(out, p, epoch)
        return p_1

    def CEL(self, input, labels):
        x_mag = torch.sqrt(torch.sum(input ** 2, dim=2)).squeeze()
        loss = nn.CrossEntropyLoss()
        output = loss(x_mag, labels)
        return (output)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)
        v_mag = torch.sqrt((input ** 2).sum(dim=2, keepdim=True))
        # (batch,num_nodes,1,1?)
        zero = Variable(torch.zeros(1)).cuda()  # 为什么把0这么写
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1) ** 2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1) ** 2
        loss_lambda = 0.5
        T_c = target
        # print(target.size()) 128,10
        L_c = T_c * max_l + loss_lambda * (1 - T_c) * max_r  # 为什么T_c-1不行，原来是0的地方变成了-1，我们需要1
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()
        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        v_mag = torch.sqrt((input ** 2).sum(dim=2))
        # get the length of capsule outputs

        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data  # 不会autograd
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            input_batch = input[batch_idx]
            batch_masked = Variable(torch.zeros(input_batch.size())).cuda()
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            # 保留max位置上的数据
            all_masked[batch_idx] = batch_masked

        masked = torch.stack(all_masked, dim=0)
        masked = masked.view(input.size(0), -1)  # batch*160?
        # 有更好的mask方法
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))  # why sigmoid here
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)
        # image0-1?
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
                # handle two_channels image 感觉多余
            else:
                output_image = output.data.cpu()
            utils.save_image(output_image, 'reconstruction.png')
        self.reconstructed_image_count += 1

        error = (output - images).view(output.size(0), -1)
        error = error ** 2
        error = torch.sum(error, dim=1) * 0.001
        # print(error.size()) error([128])

        if size_average:
            error = error.mean()

        return error

    def spread_loss(self, input, labels, epoch):
        batch_size = input.size(0)
        v_mag = input
        zero = Variable(torch.zeros(1)).cuda()
        m = 0.2
        if epoch <= 20:
            m = 0.2 + 0.035 * epoch
        else:
            m = 0.9
        m = Variable(torch.ones(1) * m).cuda()
        idx = torch.arange(batch_size)
        v_label = v_mag[idx, labels].reshape(batch_size, 1)
        loss = torch.max(v_mag - v_label + m, zero) ** 2
        loss[idx, labels] = 0
        loss_all = loss.sum(dim=1)
        return loss_all.mean()

    def set_seed(self, seed):
        torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
        torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
        torch.backends.cudnn.deterministic = True  # cudnn
        np.random.seed(seed)  # numpy
        random.seed(seed)  # random and transforms


import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from tensorboardX import SummaryWriter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 185
batch_size = 128
learning_rate = 0.001

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    # transforms.ToTensor()
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
])

train_dataset = torchvision.datasets.CIFAR10(root='../input/cifar10-torch', train=True, transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../input/cifar10-torch', train=False, transform=transform,
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = DenseCaps_net().to(device)  # .half()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1)
total_p = sum([param.nelement() for param in model.parameters()])
print('Number of params: %.2fM' % (total_p / 1e6))


def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    # for i in range(batch_size):
    #     x_one_hot[i,x[i]]=1.0
    idx = torch.arange(batch_size)
    x_one_hot[idx, x] = 1
    return x_one_hot


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accracy(pre, labels):
    # print(pre.size())
    x = pre
    predicted = x.data.max(dim=1)[1].cpu()  # 可以放进cpu节约内存
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct / total


def train(epoch):
    model.train()
    total_step = len(train_loader)
    current_lr = learning_rate
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)  # .half()
        # labels_v=to_one_hot(labels,10).to(device)

        outputs = model(images, epoch)
        # loss=model.margin_loss(outputs,labels_v)+model.reconstruction_loss(images,outputs)
        loss = model.spread_loss(outputs, labels.to(device), epoch)  # + model.reconstruction_loss(images, outputs)
        # loss=model.CEL(outputs,labels.to(device))+model.reconstruction_loss(images,outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accra = accracy(outputs, labels)
        print('Epoch:{}, step[{}/{}] Loss:{:.4f} accracy:{:.2f}%'
              .format(epoch, i, total_step, loss.item(), accra * 100))


def test(epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)  # .half()
            # labels_v=to_one_hot(labels,10).to(device)
            labels = labels.to(device)
            outputs = model(images, epoch)
            _, predictde = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predictde == labels).sum().item()
            accuracy = correct / total

        print('accuracy of the model on the test images:{}%'
              .format(100 * accuracy))

        return accuracy


temp = 0
for i in range(num_epochs):
    train(i)
    accuracy = test(i)
    scheduler.step(accuracy)
    test_error = str(1 - accuracy)
    with open('error_graph.txt', 'a') as f:
        f.write(str(i))
        f.write(' ')
        f.write(test_error)
        if i < num_epochs:
            f.write('\n')
    if accuracy > temp:
        print('early stop')
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, 'model_state.pth')
        # break
        temp = accuracy

total_p = sum([param.nelement() for param in model.parameters()])
print('Number of params: %.2fM' % (total_p / 1e6))