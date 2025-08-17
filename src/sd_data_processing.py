import os
import shutil
import pandas as pd
import json
from tqdm import tqdm

class DataPreprocessing:
    def __init__(self, root_dir = "../../Downloads/014.동화 삽화 생성 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터", img_dir = "./datasets/fairyTale/images", json_dir = "./datasets/fairyTale/labels"):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.json_dir = json_dir
        self.fairy_tales = {"image": [], "text": []}

    def moveFairyTaleData(self):
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir, exist_ok=True)

        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir, exist_ok=True)

        dir_list = os.listdir(self.root_dir)
        for dir in tqdm(dir_list):
            file_list = os.listdir(os.path.join(self.root_dir, dir))
            for file in file_list:
                if file.endswith('json'):
                    shutil.move(os.path.join(self.root_dir, dir, file), os.path.join(self.json_dir, file))
                else:
                    shutil.move(os.path.join(self.root_dir, dir, file), os.path.join(self.img_dir, file))

    
    def makeFairyTaleCsv(self):
        text_list = os.listdir(self.json_dir)

        for text in tqdm(text_list):
            with open(self.json_dir + "/" + text, 'r', encoding="utf-8-sig") as f:
                data = json.load(f)
                file_name = data['imageInfo'][0]['srcImageFile']
                name, ext = file_name.split('.')
                img_name = name + "." + ext.lower()
                self.fairy_tales['image'].append(img_name)
                self.fairy_tales['text'].append(data['imageInfo'][0]['imageCaptionInfo']['imageCaption'])

        ft = pd.DataFrame(self.fairy_tales)
        ft.to_csv('./datasets/fairyTale/fairyTale.csv', index=False)


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.moveFairyTaleData()
    dp.makeFairyTaleCsv()