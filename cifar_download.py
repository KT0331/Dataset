import torch 
from torchvision import datasets
from torchvision import transforms


def main():
    # batchsz = 32

    # cifar_train = datasets.CIFAR10(root='./',
    #     transform=transforms.Compose([
    #         transforms.Resize((32,32)),
    #         transforms.ToTensor()]),
    #     download=True)

    cifar_train = datasets.CIFAR100(root='./',
        transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()]),
        download=True)

    # cifar_train = datasets.CIFAR10(root='./cifar10/train/',
    #     train=True,
    #     transform=transforms.Compose([
    #         transforms.Resize((32,32)),
    #         transforms.ToTensor()]),
    #     download=True)

    # cifar_train = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)

    # cifar_test = datasets.CIFAR10(root='./cifar10/test/',
        # train=False,
        # transform=transforms.Compose([
            # transforms.Resize((32,32)),
            # transforms.ToTensor()]),
        # download=True)

    # cifar_test = DataLoader(cifar_test,batch_size=batchsz,shuffle=True)


if __name__ == "__main__":
    main()
