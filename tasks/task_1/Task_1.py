#!/usr/bin/env python
# coding: utf-8

#необходимые таблицы 
#для признаков
#'P_CDQ', 'P_CBC', 'P_FASTQX', 'P_OHXDEN', 'P_SLQ', 'P_DEMO', 'P_PAQ', 'P_OHQ', 'P_BPXO', 'P_BPQ', 'P_BMX', 'P_RXQASA'
#для лейблов
#'P_GHB.XPT', 'P_GLU.XPT'

import catboost
import json
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

# Создаем папки для вывода данных
if not os.path.isdir('./images'):
    os.mkdir('./images')
if not os.path.isdir('./predictions'):
    os.mkdir('./predictions')

# Загрузим описания признаков таблиц

with open('./source/features_descr_dict.json') as json_file:
    features_descr_dict = json.load(json_file)

threshold_GH = 6.5 # % or higher
threshold_GLU = 126 # mg/dl or higher

illness_criteria_explanation = f"""
В качестве признаков для определения, болен ли человек диабетом, используем показатели:
    
1) Гликированный гемоглобин. Пороговое значение для определения, болен ли человек диабетом, берем равным {threshold_GH} %. Если гликированный гемоглобин больше или равен {threshold_GH} %, считаем, что человек болен.

2) Глюкоза натощак. Пороговое значение для определения, болен ли человек диабетом, берем равным {threshold_GLU} мг/дл. Если натощак уровень глюкозы в крови больше или равен {threshold_GLU} мг/дл, считаем, что человек болен.

Источники: 
1. Executive summary: Standards of medical care in diabetes--2010. Diabetes Care. 2010 Jan;
33 Suppl 1(Suppl 1):S4-10. doi: 10.2337/dc10-S004. PMID: 20042774; PMCID: PMC2797389.
2. American Diabetes Association. https://www.diabetes.org/a1c/diagnosis.
"""

# Для определения лейблов модели нам нужны данные из таблицы P_GHB и P_GLU

label_df = pd.merge(pd.read_sas('./tables/P_GHB.XPT'), pd.read_sas('./tables/P_GLU.XPT')[['SEQN', 'LBXGLU']], how='outer', on='SEQN')

print(illness_criteria_explanation)

tables_with_predictors_df = pd.DataFrame(columns=['SEQN'])
tables_with_predictors = ['P_CDQ', 'P_CBC', 'P_FASTQX', 'P_OHXDEN', 
                          'P_SLQ', 'P_DEMO', 'P_PAQ', 'P_OHQ', 
                          'P_BPXO', 'P_BPQ', 'P_BMX', 'P_RXQASA']

for table in tables_with_predictors:
    tables_with_predictors_df = pd.merge(pd.read_sas(f'./tables/{table}.XPT'), tables_with_predictors_df,
                                     on='SEQN', how='outer')

print('Набор данных собран\n')

columns_to_select = ['SEQN', 'BPQ100D', 'RIDAGEYR', 'BMXWAIST', 'LBXMPSI', 'RXQ515', 'BMXBMI',
                   'LBXMCHSI', 'PHDSESN', 'LBXWBCSI', 'BPXOPLS2', 'CDQ001', 'BMXLEG',
                   'RXQ510', 'LBXMC', 'LBXMOPCT', 'CDQ010', 'BPXOPLS1', 'OHQ850', 'OHQ620',
                   'OHQ640', 'LBXMCVSI', 'DMDEDUC2', 'OHQ612', 'PAQ650', 'LBDNENO',
                   'OHQ835', 'OHQ860', 'LBXNEPCT', 'BPQ050A', 'LBXHGB', 'LBXRBCSI',
                   'DMDYRUSZ', 'LBDLYMNO', 'WTINTPRP', 'RIAGENDR', 'BMXARMC', 'BPXODI2',
                   'BPQ070', 'BPXODI3', 'WTMECPRP', 'BPXOPLS3', 'RIDRETH1', 'SLQ120',
                   'BMXWT', 'LBXHCT', 'PAD660', 'OHX31TC', 'OHX28CTC', 'BPD035',
                   'RIDRETH3']

full_df = tables_with_predictors_df[columns_to_select].copy()

label_df = pd.merge(full_df, label_df, how='left', on='SEQN')[['LBXGH', 'LBXGLU', 'SEQN']]
full_df.drop('SEQN', axis=1, inplace=True)

# заполняем NA значения в столбцах с категориальной переменной

cat_features = []
float_features = []

for col in full_df:
    if full_df[col].dtype == 'float':
        if len(set(full_df[col].unique()).difference({1,2,3,4,5,6,7,8,9})) == 0:
            full_df[col] = full_df[col].astype(int)
            cat_features += [col]
            full_df[col].fillna(0, inplace=True)
        else:
            float_features += [col]
    elif full_df[col].dtype == 'object':
        cat_features += [col]
        mode_type = type(full_df[col].mode().values[0])
        if mode_type == bytes:
            full_df[col].fillna(bytes('NA', 'utf-8'), inplace=True)
        else:
            full_df[col].fillna('NA', inplace=True)

print('Данные преобразованы\n')

# загружаем предсохраненные модели
filename = './source/full_model'
filename_male = './source/full_model_male'
filename_female = './source/full_model_female'

clf_full = pickle.load(open(filename, 'rb'))
clf_full_male = pickle.load(open(filename_male, 'rb'))
clf_full_female = pickle.load(open(filename_female, 'rb'))

# формируем тестовые наборы
test_pool = catboost.Pool(
    full_df,
    cat_features=cat_features
)
sorted_feature_importance_full = [features_descr_dict[col]['label'] 
                                  for col in full_df.columns[clf_full.feature_importances_.argsort()[-5:]]]
print('Топ-5 признаков по значимости для полного набора данных\n - ', end='')
print('\n - '.join(sorted_feature_importance_full), end='\n\n')

full_df_male = full_df[full_df.RIAGENDR == 1].copy()
test_pool_male = catboost.Pool(
    full_df_male,
    cat_features=cat_features
)
sorted_feature_importance_male = [features_descr_dict[col]['label'] 
                                  for col in full_df.columns[clf_full_male.feature_importances_.argsort()[-5:]]]
print('Топ-5 признаков по значимости для мужчин\n - ', end='')
print('\n - '.join(sorted_feature_importance_male), end='\n\n')

full_df_female = full_df[full_df.RIAGENDR == 2].copy()
test_pool_female = catboost.Pool(
    full_df_female,
    cat_features=cat_features
)
sorted_feature_importance_female = [features_descr_dict[col]['label'] 
                                  for col in full_df.columns[clf_full_female.feature_importances_.argsort()[-5:]]]
print('Топ-5 признаков по значимости для женщин\n - ', end='')
print('\n - '.join(sorted_feature_importance_female), end='\n')

label_df.fillna(0, inplace=True)
label_df['Is_diabetic'] = label_df.apply(lambda x: 1 if (x.LBXGH >= threshold_GH) or (x.LBXGLU >= threshold_GLU) 
                                         else 0, axis=1)
print('\nМетки болен/здоров назначены\n')

predictions_df = label_df[['SEQN']].copy()
predictions_df['score'] = clf_full.predict_proba(test_pool, thread_count=8)[:,1]
predictions_df.to_csv('./predictions/predictions_full.csv', index=False)
print('Файл с предсказанием вероятности наличия у человека сахарного диабета для полного набора данных сохранен')

predictions_df_male = pd.merge(full_df_male, label_df, how='left', left_index=True, right_index=True)[['SEQN']]
predictions_df_male['score'] = clf_full_male.predict_proba(test_pool_male, thread_count=8)[:,1]
predictions_df_male.to_csv('./predictions/predictions_male.csv', index=False)
print('Файл с предсказанием вероятности наличия у человека сахарного диабета для мужчин сохранен')


predictions_df_female = pd.merge(full_df_female, label_df, how='left', left_index=True, right_index=True)[['SEQN']]
predictions_df_female['score'] = clf_full_female.predict_proba(test_pool_female, thread_count=8)[:,1]
predictions_df_female.to_csv('./predictions/predictions_female.csv', index=False)
print('Файл с предсказанием вероятности наличия у человека сахарного диабета для женщин сохранен\n')


models = [clf_full, clf_full_male, clf_full_female]
test_pools = [test_pool, test_pool_male, test_pool_female] 
labels = [label_df['Is_diabetic'], 
          pd.merge(full_df_male, label_df, how='left', left_index=True, right_index=True)['Is_diabetic'],
          pd.merge(full_df_female, label_df, how='left', left_index=True, right_index=True)['Is_diabetic']]
titles = ['весь набор данных', 'мужчины', 'женщины']
suffixes = ['full', 'male', 'female']

for clf_full, test_pool, y_test, title, suffix  in zip(models, test_pools, labels, titles, suffixes):
    test_full_pred = clf_full.predict_proba(test_pool)[:,1]

    plt.figure(figsize=[10,8])

    df = pd.DataFrame({'probPos':test_full_pred, 'target': y_test})
    sns.histplot(df[df.target==0].probPos, bins=50, kde=True, color='g', log_scale=True,
             alpha=.4, label='Заболевание отсутствует')
    sns.histplot(df[df.target==1].probPos, bins=50, kde=True,
             alpha=.55, color='r',  label='Заболевание присутствует', log_scale=True)
    plt.axvline(.5, color='b', linestyle='--', label='Порог')
    plt.xlim([0,1])
    plt.ylim([0,1500])
    plt.title(f'Распределение вероятностей ({title})', size=16, pad=10)
    plt.xlabel('Вероятность наличия заболевания (предсказанная)', size=14, labelpad=8)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.ylabel('Количество объектов', size=14, labelpad=8)
    plt.legend(loc="best", fontsize=14)
    plt.savefig(f'./images/distribution_{suffix}.png', facecolor='w', bbox_inches='tight')
    print(f'Изображение с распределением вероятности наличия у человека сахарного диабета ({title}) сохранено')
print('')
