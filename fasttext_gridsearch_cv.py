import csv
import os
import re
import fasttext
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import copy
import tempfile
import shutil
from sklearn.metrics import *


# 网格搜索+交叉验证
# 输入：训练数据帧，要搜索的参数（字典形式），k-fold交叉验证

def my_gridsearch_cv_mse(df, param_grid, k_fold=10):
    n_classes = len(np.unique(df[1]))
    print('n_classes', n_classes)

    kf = KFold(n_splits=k_fold)  # k折交叉验证

    params_combination = get_gridsearch_params(param_grid)  # 获取参数的各种排列组合

    best_score = 10000.0  # mse越小越好，故设定较大的初值
    best_params = dict()
    for params in params_combination:
        avg_score = get_k_fold_scores(df, params, kf, n_classes)
        if avg_score < best_score:  # mse越小越好
            best_score = avg_score
            best_params = copy.deepcopy(params)  # 拷贝参数
    return best_score, best_params


# 将各个参数的取值进行排列组合
def get_gridsearch_params(param_grid):
    params_combination = [dict()]  # 存放所有可能的参数组合
    for k, v_list in param_grid.items():
        tmp = [{k: v} for v in v_list]
        n = len(params_combination)
        copy_params = [copy.deepcopy(params_combination) for _ in range(len(tmp))]
        params_combination = sum(copy_params, [])
        _ = [params_combination[i * n + k].update(tmp[i]) for k in range(n) for i in range(len(tmp))]
    return params_combination


# 使用k折交叉验证，得到最后的score，保存最佳score以及其对应的那组参数
# 输入分别是训练数据帧，要搜索的参数，用于交叉验证的KFold对象，分类数
def get_k_fold_scores(df, params, kf, n_classes):
    metric_score = 0.0

    for train_idx, val_idx in kf.split(df):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        tmpdir = tempfile.mkdtemp()  # 为交叉验证新建一个临时目录
        tmp_train_file = tmpdir + '/train.txt'
        df_train.to_csv(tmp_train_file, sep='\t', quoting=csv.QUOTE_NONE, index=False, header=None,
                        encoding='UTF-8')  # 不要表头

        fast_model = fasttext.train_supervised(tmp_train_file, label_prefix='__label__', thread=3,
                                               **params)  # 使用fastText训练，传入参数

        # print(df_val[0].tolist()) #['__label__4', '__label__2', '__label__5', '__label__5', ...]
        # print(df_val[1].tolist())  #['text1', 'text2',...]
        # 用训练好的模型做评估预测
        predicted = fast_model.predict(df_val[1].tolist())
        y_val_pred = [int(label[0][-1:]) for label in predicted[0]]  # label[0]  __label__0 （预测的label）
        y_val = [int(cls[-1:]) for cls in df_val[0]]  # 验证集的ground truth

        score = mean_squared_error(y_val, y_val_pred)  # 调用sklearn的mean_squared_error计算MSE
        metric_score += score  # 累计在整个交叉验证集上的MSE得分
        shutil.rmtree(tmpdir, ignore_errors=True)  # 删除临时训练数据文件

    print('平均MSE:', metric_score / kf.n_splits)  # 计算在整个交叉验证集上平均分
    return metric_score / kf.n_splits


def string_formatting(string):
    """将文本转换为小写，并在标点符号前添加空格。"""
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r" \1 ",
                    string)  # 在标点符号前添加空格
    return string


# 可供选择的参数列表
"""
tuned_parameters = {
    'lr': [0.1, 0.05],
    'epoch': [15, 20, 25, 30],
    'dim': [50, 100, 150, 200],
    'wordNgrams': [2, 3],
}
"""

# 经过实验选择的效果较好的一组超参数
tuned_parameters = {
    'lr': [0.1],
    'epoch': [30],
    'dim': [150],
    'wordNgrams': [2],
}

if __name__ == '__main__':
    reviews_path = 'D:/PycharmProjects/yelp_sa/HomeworkData.csv'  # 评论数据集路径
    tagged_reviews_path = 'D:/PycharmProjects/yelp_sa/tagged_review_texts.txt'  # 用于保存生成的txt

    # 数据集存在部分UTF-8字符
    with open(reviews_path, "r", encoding='utf-8') as input_, open(tagged_reviews_path, "w",
                                                                   encoding='utf-8') as tagged_output:
        reader = csv.reader(input_)
        fieldnames = next(reader)  # 获取数据的第一列，作为后续要转为字典的键名
        # print(fieldnames)
        csv_reader = csv.DictReader(input_,
                                    fieldnames=fieldnames)  # 以list的形式存放键名
        for row in csv_reader:
            d = {}
            for k, v in row.items():
                d[k] = v
            # print(d)

            rating = d['stars']
            text = d['text'].replace("\n", " ")
            text = string_formatting(text)

            fasttext_line = "__label__{}\t{}".format(int(float(rating)), text)

            tagged_output.write(fasttext_line + "\n")

    # 文件大小（MB）
    file_size = os.stat(tagged_reviews_path).st_size / 1e+6
    print(f'tagged_review_texts, file size is: {file_size} MB \n')

    df_ = pd.read_csv(tagged_reviews_path, encoding='UTF-8', sep='\t', header=None, index_col=False, usecols=[0, 1],
                      warn_bad_lines=True, error_bad_lines=False)  # 将txt当作csv读入，获得一个数据帧，仅有label和text两列

    print(df_.head())
    print(df_.shape)
    best_score_mse, best_params_mse = my_gridsearch_cv_mse(df_, tuned_parameters, k_fold=10)
    print('best_score (MSE)', best_score_mse)
    print('best_params', best_params_mse)
