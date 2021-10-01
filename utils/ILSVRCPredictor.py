import numpy as np


class ILSVRCPredictor():
    """
    ILSVRCデータに対するモデルの出力からクラスラベルを求めて返却

    Attributes
    ----------
    class_index : dictionary
        クラスindexとラベルを対応させた辞書型変数

    """

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        """
        確率が最大のILSVRCのラベル名を取得する

        Parameters
        ----------
        out : torch.Size([1, 1000])
            Netからの出力

        Returns
        ----------
        predicted_label_name : str
            予測確率が最大のラベル名
        """
        maxid = np.argmax(out.detach().numpy()
                          )  # Netからの出力を切り離す + numpy配列へ変換 その後最大のindexをmaxidへ代入
        predicted_label_name = self.class_index[str(maxid)][1]
        return predicted_label_name


if __name__ == '__main__':
    """
    使用例
    実行方法: python3 ILSVRCPredictor.py
    """
    import json
    from PIL import Image
    from torchvision import models
    from ImageTransform import ImageTransform

    # サンプルとして学習済みVGGを用意
    use_pretrained = True
    net = models.vgg16(pretrained=use_pretrained)
    net.eval()  # モデルのインスタンスに対して推論モードに設定

    # 画像を用意
    img_path = '../data/golden_retriever_01.jpg'
    img = Image.open(img_path)

    # 画像をPyTorchのTensor型へ変換
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(size, mean, std)
    img_transformed = transform(img, phase='train')
    inputs = img_transformed.unsqueeze_(0)  # バッチ次元を先頭に追加

    # 推論結果を取得
    out = net(inputs)

    # jsonファイルからindexとクラスラベルが対応づけられた辞書を辞書型変数としてロードする
    json_file_path = '../data/imagenet_class_index.json'
    class_index = json.load(open(json_file_path, 'r'))
    # 辞書型変数を引数にしてインスタンスの作成
    predictor = ILSVRCPredictor(class_index)

    # 推論結果のクラスを出力
    result = predictor.predict_max(out)
    print(result)
