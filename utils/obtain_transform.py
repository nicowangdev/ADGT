import torchvision.transforms as transforms
def obtain_transform(dataset_name):
    if dataset_name== 'C10' or dataset_name == 'C100':
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset_name == 'Flower102':
        size = 224
        padding=32
        transform_train = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([transforms.Resize(size + padding),
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
    elif dataset_name =='RestrictedImageNet':
        size = 224
        padding = 32
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([transforms.Resize(size + padding),
                                             transforms.CenterCrop(size),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])
    else:
        transform_train = transform_test = transforms.ToTensor()
    return transform_train,transform_test