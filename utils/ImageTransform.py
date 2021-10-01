# import
import random
import numpy as np
# import: pytorch
import torch
from torchvision import transforms

# 再現設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
# 再現設定: GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class ImageTransform():
    """
    [numpyスタイルでdocstringを書く]
    画像の前処理クラス
    trainとvalidationで異なる動作をする

    Attributes
    ----------
    resize : int
        リサイズ後の画像サイズ
    mean : (R, G, B)
        各色チャネルの平均値
    std : (R, G, B)
        各色チャネルの標準偏差
    """

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                # scaleの範囲でランダムに切り取られた後にresizeで指定したサイズに拡大or縮小
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),  # ランダムに左右反転
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),  # リサイズ
                transforms.CenterCrop(resize),  # 中央を切り抜き
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(mean, std)  # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        """
        何もメソッドを指定していない時に呼び出される

        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定
        """
        return self.data_transform[phase](img)


if __name__ == '__main__':
    """
    使用例
    実行方法: python3 ImageTransform.py
    """

    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt

    img_path = '../data/golden_retriever_01.jpg'
    # img = cv2.imread(img_path)
    # cv2.imwrite('../output/input.jpg', img)

    img = Image.open(img_path)

    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = ImageTransform(size, mean, std)
    img_transformed = transform(img, phase='train')

    # PyTorch形式からOpenCV形式への変換: (color, height, width)を(height, width, color)へ変換
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    # 0-1の範囲外の値があれば指定してた最小値、最大値で置換する
    img_transformed = np.clip(img_transformed, 0, 1)

    # 1. matplot.pyplot形式で保存
    # plt.imshow(img_transformed) # matplotlibだと0-1のままでも自動で補完してくれるので表示が可能
    # plt.savefig('../output/out_plt.png')

    # 2. OpenCV形式で保存
    # OpenCVだと0-255の範囲に戻して画像の値として扱えるように変換する必要あり
    img_transformed = img_transformed*255
    # PILからOpenCVへ変換: RGBからBGRへ変換
    img_transformed = cv2.cvtColor(img_transformed, cv2.COLOR_RGB2BGR)
    print(img_transformed)
    cv2.imwrite('../output/out_cv2.png', img_transformed)
