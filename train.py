import datetime
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.loss import YOLOLoss, get_lr_scheduler, set_optimizer_lr
from nets.yolo import YoloBody
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate


def get_ann_info(root, data_name='train'):
    path = os.path.join(root, data_name)
    json_path = os.path.join(path, '_annotations.coco.json')
    with open(json_path, 'r') as f:
        ann_json = json.load(f)

    data = []
    for img in tqdm(ann_json['images']):
        id = img['id']
        file_name = os.path.join(path, img['file_name'])
        t = []
        for ann in ann_json['annotations']:
            if ann['image_id'] == id:
                bbox = ann['bbox']
                bbox = bbox[:2] + [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                d = list(map(lambda x: str(int(x)), bbox + [ann['category_id'] - 1]))
                t.append(','.join(d))

        data.append(f"{file_name} {' '.join(t)}")

    return data


if __name__ == "__main__":
    input_shape = [640, 640]
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7

    epochs = 300
    batch_size = 12
    Init_lr = 0.01
    Min_lr = Init_lr * 0.01
    save_dir = 'logs'
    eval_period = 10
    num_workers = 12

    # classes
    class_names, num_classes = ['Boom', 'Groep bomen', 'Rij bomen'], 3
    anchors = np.array([[12, 16], [19, 36], [40, 28], [36, 75],
                        [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    num_anchors = len(anchors)

    model = YoloBody(anchors_mask, num_classes)
    model.train()
    model = model.cuda()
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir)

    # 准备数据
    data_root = 'datasets/LOD/'
    train_lines = get_ann_info(data_root, 'train')
    num_train = len(train_lines)
    val_lines = get_ann_info(data_root, 'valid')
    num_val = len(val_lines)

    # 学习率
    lr_limit_max = 1e-3
    lr_limit_min = 3e-4
    Init_lr_fit = min(max(batch_size / 64 * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / 64 * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 初始化参数
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = optim.Adam(pg0, Init_lr_fit, betas=(0.937, 0.999))
    optimizer.add_param_group({"params": pg1, "5e-4": 5e-4})
    optimizer.add_param_group({"params": pg2})
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, 0)
    lr_scheduler_func = get_lr_scheduler("cos", Init_lr_fit, Min_lr_fit, epochs)

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask,
                                epoch_length=epochs,
                                mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob,
                                mixup_prob=mixup_prob, train=True,
                                special_aug_ratio=special_aug_ratio)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask,
                              epoch_length=epochs,
                              mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                              train=False, special_aug_ratio=0)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                     num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate,
                     )
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size,
                         num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate,
                         )

    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names,
                                 num_classes, val_lines, log_dir,
                                 eval_flag=True, period=eval_period)

    for epoch in range(epochs):
        gen.dataset.epoch_now = epoch
        gen_val.dataset.epoch_now = epoch

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = 0
        val_loss = 0

        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'epochs {epoch + 1}/{epochs}')
        model.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            images = images.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()

            outputs = model(images)
            loss_value = yolo_loss(outputs, targets, images)
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            pbar.set_postfix(**{'loss': loss / (iteration + 1)})
            pbar.update(1)
        pbar.close()

        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'epochs {epoch + 1}/{epochs}')
        model.eval()
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = images.cuda()
                targets = targets.cuda()
                optimizer.zero_grad()
                outputs = model(images)
                loss_value = yolo_loss(outputs, targets, images)

            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model)
        print('epochs:' + str(epoch + 1) + '/' + str(epochs))
        print(f'Total Loss: {loss / epoch_step:.3f}|| Val Loss: {val_loss / epoch_step_val:.3f} ')

        save_state_dict = model.state_dict()
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(
                loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))

    loss_history.writer.close()
