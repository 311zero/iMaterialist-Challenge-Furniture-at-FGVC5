import torch
from torch.autograd import Variable
from tqdm import tqdm

use_gpu = torch.cuda.is_available()


class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count
        self.total_value

    # @property广泛应用在类的定义中，可以让调用者写出简短的代码，同时保证对参数进行必要的检查，这样，程序运行时就减少了出错的可能性。
    @property  # decorator,get,Python内置的@property装饰器就是负责把一个方法变成属性调用的
    def value(self):
        if self.count:
            return self.total_value / self.count  # 求平均值
        else:
            return float("inf")  # 正无穷

    def __str__(self):
        return str(self.value)


def predict(model, dataloader):
    all_labels = []
    all_outputs = []
    model.eval()  # 框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值

    pbar = tqdm(dataloader, total=len(dataloader))
    for inputs, labels in pbar:
        all_labels.append(labels)

        inputs = Variable(inputs, volatile=True)
        if use_gpu:
            inputs = inputs.cuda()

        outputs = model(inputs)
        all_outputs.append(outputs.data.cpu())

    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    if use_gpu:
        all_labels = all_labels.cuda()
        all_outputs = all_outputs.cuda()

    return all_labels, all_outputs


def safe_stack_2array(a, b, dim=0):  # dim = 0
    if a is None:
        return b
    return torch.stack((a, b), dim=dim)


def predict_tta(model, dataloaders):
    prediction = None
    lx = None
    for dataloader in dataloaders:
        lx, px = predict(model, dataloader)
        prediction = safe_stack_2array(prediction, px, dim=-1)

    return lx, prediction
