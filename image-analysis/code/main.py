import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import timm
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import os
import sys
import time
import json
import shutil

PATH_DATA = 'C:/Users/DELL/Desktop/data'

categories = os.listdir(os.path.join(PATH_DATA, 'classified'))


def data_preprocess():
    global PATH_DATA, categories

    data_chart = pd.read_csv(os.path.join(PATH_DATA, 'data.csv'))
    img_cls = {cate: [img for img in os.listdir(os.path.join(PATH_DATA, 'classified', cate))] for cate in categories}

    for i in range(1, 4):  # read in labels
        col = f'img{i}_cls'
        data_chart.insert(data_chart.shape[1], col, 'None')
        img_cls_list = data_chart[col]
        for j, img in enumerate(data_chart[f'images_{i}']):
            for k, v in img_cls.items():
                if img in v:
                    data_chart.loc[j, col] = k
                    break

    data_chart.to_csv(os.path.join(PATH_DATA, 'overall.csv'))
    random_chart = data_chart.copy()
    random_chart = random_chart.sample(frac=1)

    keys = ['train', 'val', 'test']
    vals = np.array_split(random_chart, [int(0.6 * random_chart.shape[0]), int(0.8 * random_chart.shape[0])])

    charts = dict(zip(keys, vals))

    for type, chart in charts.items():
        chart.reset_index(inplace=True)
        chart.to_csv(os.path.join(PATH_DATA, f'{type}.csv'))

        dir = os.path.join(PATH_DATA, type)
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.mkdir(dir)
        for cls in categories:
            os.mkdir(os.path.join(dir, cls))

        for i in range(1, 4):
            img_list = f'images_{i}'
            for j, file in enumerate(chart[img_list]):
                if file is None:
                    continue
                cls = str(chart[f'img{i}_cls'][j])
                if cls == 'None':
                    continue
                image = cv2.imread(os.path.join(PATH_DATA, 'images', str(file)))
                cv2.imwrite(os.path.join(dir, (chart.loc[j, f'img{i}_cls']), str(file)), image)


def data_hotel_cls(batch_size, num_workers=0, data_transform=None):
    global PATH_DATA

    if data_transform is None:
        data_transform = {
            "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "val": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            "test": transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor()]),
        }

    # image_path = os.path.join(PATH_DATA, 'images')
    train_dataset = datasets.ImageFolder(root=os.path.join(PATH_DATA, 'train'),
                                         transform=data_transform['train'])
    val_dataset = datasets.ImageFolder(root=os.path.join(PATH_DATA, 'val'),
                                       transform=data_transform['val'])
    test_dataset = datasets.ImageFolder(root=os.path.join(PATH_DATA, 'test'),
                                       transform=data_transform['test'])
    cls_list = train_dataset.class_to_idx
    cls_dict = dict((val, key) for key, val in cls_list.items())
    json_str = json.dumps(cls_dict, indent=4)
    with open(os.path.join(PATH_DATA, 'cls.json'), 'w') as json_file:
        json_file.write(json_str)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def train(net, train_loader, val_loader, loss_function, optimizer, scheduler, device, epoch_num, save_dir=None):
    timseq = [time.time()]

    train_num = len(train_loader.dataset)
    val_num = len(val_loader.dataset)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.__next__()  # not the `.next()` but the `.__next__()`

    train_loss, val_accuracy = [], []
    best_acc = 0.0
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        # -------------------------------------------- train --------------------------------------------------
        net.train()  # set as train mode, keep the dropout
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)  # 进度条
        for step, data in enumerate(train_bar):
            inputs, labels = data  # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad()  # zero the parameter gradients
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()  # forward + backward + optimize
            optimizer.step()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epoch_num, loss)

            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image.to(device))  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label.to(device)).sum().item() / val_label.size(0)

                    print('[epoch: %d, step: %5d] last_500_loss: %.3f  test_accuracy: %.3f'
                          % (epoch + 1, step + 1, running_loss / 500, accuracy))

            running_loss += loss.item()

        train_loss.append(running_loss / train_num)

        if scheduler is not None:
            scheduler.step()

        # -------------------------------------------- validate ----------------------------------------------
        net.eval()  # call back dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]  # 提出最大概率
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()  # val_labels.to(device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epoch_num)

        # Record processing
        val_accuracy.append(acc / val_num)
        timseq.append(time.time())
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f cost time: %.2f' %
              (epoch + 1, train_loss[-1], val_accuracy[-1], timseq[-1] - timseq[-2]))

        # save the best model
        if val_accuracy[-1] > best_acc:
            best_acc = val_accuracy[-1]
            torch.save(net.state_dict(), os.path.join(save_dir, 'weight', "best.pth"))

    # ------------------------------------ save result ----------------------------------------------------
    # save weight
    torch.save(net.state_dict(), os.path.join(save_dir, 'weight', 'last.pth'))  # weight

    # save matplot pics
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), tight_layout=True)
    axes[0].plot(train_loss, label="train loss", ls='-', color='black')
    axes[1].plot(val_accuracy, label="val acc", ls='-', color='red')
    fig.savefig(os.path.join(save_dir, 'result.jpg'))  # pic
    plt.show()

    # save csv chart
    df = pd.DataFrame(np.array([np.arange(0, epoch_num), train_loss, val_accuracy]).T,
                      index=None, columns=['epoch', 'train_loss', 'val_accuracy'])
    df.to_csv(os.path.join(save_dir, 'result.csv'), index=False)  # csv

    # print optimizer
    print("Hyper-parameter:")
    print("optim:", optimizer, "\nepoch_num:", epoch_num)
    print("run time: %.2f s" % (timseq[-1] - timseq[0]))
    print('Finished Training')


def generate_savedir(save_name) -> str:
    # Generate save directory with 'YearMonthDate-Hour-Minute-Second' format.
    # the save file name doesn't allow the sign ':'
    str_now = time.strftime("%Y%m%d-%H-%M-%S")
    if save_name is None:
        save_name = str_now
    else:
        save_name = str_now + '-' + save_name
    save_dir = os.path.join('../output', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'weights'))

    return save_dir


def predict(device, model, weight_path, test_chart):
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    predict_chart = test_chart.copy()

    model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    json_path = os.path.join(PATH_DATA, 'cls.json')
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, 'r') as f:
        class_indict = json.load(f)

    with torch.no_grad():
        for i in range(1, 4):
            img_list = f'images_{i}'
            col = f'predict_img{i}'
            predict_chart.insert(predict_chart.shape[1], col, 'None')
            for j, file in enumerate(chart[img_list]):
                if file is None:
                    continue
                image = Image.open(os.path.join(PATH_DATA, 'images', str(file)))
                image = data_transform(image)
                image = torch.unsqueeze(image, dim=0)

                output = torch.squeeze(model(image.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cls = torch.argmax(predict).numpy()

                predict_chart.loc[j, col] = class_indict[str(predict_cls)]

    predict_chart.to_csv(os.path.join(PATH_DATA, 'predict.csv'))


if __name__ == '__main__':
    # hyper-param
    batch_size = 32
    num_epoch = 10
    learning_rate = 0.01

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = data_hotel_cls(batch_size=batch_size)

    # Xception
    model = timm.create_model('xception', pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, len(categories))
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # Stochastic Gradient Descent

    save_dir = generate_savedir('Xception_hotel_CLS')
    train(model, train_loader, val_loader, loss_func, optimizer, None, device, num_epoch, save_dir)

    test_chart = pd.read_csv(os.path.join(PATH_DATA, 'test.csv'))
    predict(device, model, "../output/20230603-16-35-48-Xception_hotel_CLS/weight/last.pth",
            test_chart)

