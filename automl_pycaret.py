import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import nltk

# nltk.download('stopwords')
warnings.filterwarnings('ignore')
df = pd.read_csv('./HomeworkData.csv')
df.drop(columns=['review_id','date','user_id', 'business_id'])
print(df.head())

from nltk.corpus import stopwords

stop_words = stopwords.words('english')
from pycaret.nlp import *

su_1 = setup(data=df, target='text', custom_stopwords=stop_words, session_id=21)
m1 = create_model(model='lda', multi_core=False)
lda_data = assign_model(m1)
lda_data.head()
evaluate_model(m1)

#tune_model(model='lda', multi_core=True, supervised_target='stars', estimator=None, optimize='MSE', auto_fe=True, fold=10)

# Non-negative Matrix Factoring
"""
m2 = create_model(model='nmf', multi_core=True)
nmf_data = assign_model(m2)
nmf_data.head()
nmf_data.columns
nmf_data.drop(['text', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace=True)
nmf_data.head()
"""

'''
from pycaret.classification import *

pce_1 = setup(data=lda_data, target='stars', session_id=100, train_size=0.9)
compare_models()
'''

from pycaret.regression import *
lda_data.drop(['text', 'Dominant_Topic', 'Perc_Dominant_Topic'], axis=1, inplace=True)
pce_1 = setup(data=lda_data, target='stars', session_id=200, train_size=0.9)
compare_models()
