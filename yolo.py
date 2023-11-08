import colorsys
import os
import numpy as np
import torch
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, preprocess_input, resize_image)
from utils.utils_bbox import DecodeBox


class YOLO:
    def __init__(self, model_path='yolov7_weights.pth', input_shape=[640, 640], confidence=0.5,
                 nms_iou=0.3, letterbox_image=False,
                 cuda=True, **kwargs):
        self.cuda = cuda
        self.model_path = model_path
        self.input_shape = input_shape
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.class_names, self.num_classes = ['Boom', 'Groep bomen', 'Rij bomen'], 3
        self.anchors = np.array([[12, 16], [19, 36], [40, 28], [36, 75],
                                 [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]])
        self.num_anchors = len(self.anchors)
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.bbox_util = DecodeBox(self.anchors, self.num_classes,
                                   (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def get_defaults(self, n):
        if hasattr(self, n):
            return getattr(self, n)
        else:
            return "Unrecognized attribute name '" + n + "'"

    def generate(self):
        self.net = YoloBody(self.anchors_mask, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.net.to(device)
        self.net = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))

        self.net.cuda()

    # 检测图片
    def detect_image(self, image, crop=False, count=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])

        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]),
                                  self.letterbox_image)

        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        image_data = np.expand_dims(
            np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()
            #   将图像输入网络当中进行预测！
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)

            #   将预测框进行堆叠，然后进行非极大抑制
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes,
                                                         self.input_shape,
                                                         image_shape, self.letterbox_image,
                                                         conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        #   设置字体与边框厚度
        font = ImageFont.truetype(font='simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #   计数
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        #   是否进行目标的裁剪
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95,
                                subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)

        #   图像绘制
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],
                           fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
