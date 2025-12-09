import pandas as pd, numpy as np

df = pd.read_excel('test1.xlsx', sheet_name='Sheet1')

cols_num = ['heart_rate','resp_rate','temperature','sbp','dbp','map','sao2',
            'ph','lactate','hemoglobin','urine_output_ml_hr']

# 1. 同 stay 线性插值
df[cols_num] = (df.groupby('stay_id')[cols_num]
                  .apply(lambda x: x.interpolate(method='linear', limit_area='inside'))
                  .reset_index(level=0, drop=True))

# 2. 其余缺失用全局中位数
med = df[cols_num].median()
df[cols_num] = df[cols_num].fillna(med)

# 3. 血管活性药缺失视为 0
vaso = ['norepinephrine','dopamine','epinephrine','vasopressin','phenylephrine']
df[vaso] = df[vaso].fillna(0)

# 4. 体温无值时默认 36.5 ℃
df['temperature'] = df['temperature'].fillna(36.5)

# 保存
df.to_excel('test1_imputed.xlsx', index=False)
