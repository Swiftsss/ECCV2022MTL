from torchvision import transforms
def get_data_transforms(args):
    transforms_train = transforms.Compose([
        transforms.Resize([int(args.input_size*1.02), int(args.input_size*1.02)]),
        transforms.RandomCrop([args.input_size, args.input_size]),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])
    transforms_valid = transforms.Compose([
        transforms.Resize([int(args.input_size), int(args.input_size)]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std =[0.229, 0.224, 0.225])
    ])
    return transforms_train, transforms_valid