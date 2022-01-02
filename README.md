# fastText Yelp评分预测

## 需要导入的包
```
pip install fasttext
pip install sklearn
```
## 训练流程
修改`reviews_path`为数据集csv的路径，`tagged_reviews_path`为保存生成的txt文件的路径（fastText需要读入txt文件进行训练）。

执行代码：`python fasttext_gridsearch_cv.py`

由于只指定了一组超参数（之前调参后选取的结果最好的一组参数），所以GridSearch只会进行一组实验。由于设定了10折交叉验证，最后输出的MSE是十次交叉验证结果的均值。
