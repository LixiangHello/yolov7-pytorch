from PIL import Image
from yolo import YOLO
import os
from tqdm import tqdm

if __name__ == "__main__":
    dir_origin_path = "datasets/LOD/test"
    dir_save_path = "test_results"
    os.makedirs(dir_save_path, exist_ok=True)

    yolo = YOLO(model_path='logs/best_epoch_weights.pth')

    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(dir_origin_path, img_name)
            image = Image.open(image_path)
            r_image = yolo.detect_image(image)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")),
                         quality=95, subsampling=0)
