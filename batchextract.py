import pathlib
import pickle
import time

import pandas as pd
from radiomics import featureextractor


def extract_features(img, mask):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.enableAllFeatures()
    extractor.enableImageTypes(
        Original={},
        Wavelet={},
        # LoG={},
        # Logarithm={},
        # Exponential={},
        # Gradient={},
    )
    return extractor.execute(img, mask)


def batch_extract(cls: str):
    with open(f'./resources/extract_data/finding_list_{cls}.pkl', 'rb') as f:
        finding_list = pickle.load(f)

    temp_data = pathlib.Path('./resources/temp_data')
    temp_data.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame()
    for results in finding_list:  # results是一个字典
        img = str(results['img'].resolve(strict=True))
        mask = str(results['mask'].resolve(strict=True))
        label = results['label']
        stem = results['stem']

        # 特征提取
        featureVector = extract_features(img=img, mask=mask)

        featureVector['label'] = label
        featureVector['stem'] = stem

        # 将提取的特征转换为DataFrame格式
        df_new = pd.DataFrame.from_dict(featureVector.values()).T
        df_new.columns = featureVector.keys()
        df = pd.concat([df, df_new])

    # 将提取的特征结果写入文件
    df.to_pickle(temp_data / f'extract_results_{cls}.pkl')
    with pd.ExcelWriter(f'./resources/extract_data/extract_results_{cls}.xlsx') as writer:
        df.to_excel(writer, index=False)


if __name__ == '__main__':
    batch_extract('ceus')
