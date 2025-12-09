import pandas as pd
import numpy as np

# 1. 读入你已上传的补全文件
df = pd.read_excel('test1_imputed.xlsx', sheet_name='Sheet1')

# 2. 定义“正常值池”——均按 ICU 常见参考范围
normal_pool = {
    'heart_rate':      np.arange(60, 101, 1),          # 60-100 bpm
    'resp_rate':       np.arange(12, 21, 1),           # 12-20 次/分
    'temperature':     np.round(np.arange(36.0, 37.6, 0.1), 1),  # 36.0-37.5 ℃
    'sbp':             np.arange(90, 141, 1),          # 90-140 mmHg
    'dbp':             np.arange(60, 91, 1),           # 60-90 mmHg
    'map':             np.arange(70, 105, 1),          # 70-104 mmHg
    'fio2':            np.round(np.arange(0.21, 0.61, 0.01), 2), # 21%-60%
    'ph':              np.round(np.arange(7.35, 7.46, 0.01), 2),
    'pco2':            np.arange(35, 46, 1),            # 35-45 mmHg
    'po2':             np.arange(80, 101, 1),           # 80-100 mmHg
    'hco3':            np.arange(22, 27, 1),            # 22-26 mmol/L
    'lactate':         np.round(np.arange(0.5, 2.1, 0.1), 1),   # 0.5-2.0 mmol/L
    'sao2':            np.arange(94, 101, 1),            # 94-100 %
    'base_excess':     np.arange(-2, 3, 1),              # -2 ~ +2 mmol/L
    'tco2':            np.arange(23, 28, 1),             # 23-27 mmol/L
    'sodium':          np.arange(135, 146, 1),           # 135-145 mmol/L
    'potassium':       np.round(np.arange(3.5, 5.1, 0.1), 1),  # 3.5-5.0 mmol/L
    'chloride':        np.arange(98, 108, 1),            # 98-107 mmol/L
    'calcium':         np.round(np.arange(1.0, 1.31, 0.01), 2), # 1.00-1.30 mmol/L
    'hemoglobin':      np.arange(10.0, 16.1, 0.1),       # 10-16 g/dL
    'hematocrit':      np.arange(30, 47, 1),             # 30-46 %
    'norepinephrine':  np.array([0.0]),                   # 默认 0，除非已有
    'dopamine':        np.array([0.0]),
    'epinephrine':     np.array([0.0]),
    'vasopressin':     np.array([0.0]),
    'phenylephrine':   np.array([0.0]),
    'in_volume_ml_hr': np.arange(50, 201, 5),            # 50-200 mL/h
    'crystalloid_in_ml_hr': np.arange(30, 151, 5),
    'colloid_in_ml_hr': np.arange(0, 101, 5),
    'out_volume_ml_hr': np.arange(30, 101, 5),
    'urine_output_ml_hr': np.arange(30, 101, 5),         # 0.5-2 mL/kg/h 估算
    'pf_ratio':        np.arange(300, 501, 5),            # 300-500 mmHg
    'anion_gap':       np.arange(5, 13, 1),               # 5-12 mmol/L
}

# 3. 逐列填充
for col, pool in normal_pool.items():
    if col not in df.columns:
        continue
    miss_mask = df[col].isna()
    if miss_mask.sum() == 0:
        continue
    # 同一 stay_id 内随机抽取，保证波动
    def fill_norm(x):
        n_missing = x.isna().sum()
        if n_missing == 0:
            return x
        fills = np.random.choice(pool, size=n_missing, replace=True)
        x.loc[x.isna()] = fills
        return x
    df[col] = df.groupby('stay_id')[col].transform(fill_norm)

# 4. 保存
df.to_excel('test1_normal_filled.xlsx', index=False)
print('✅ 已全部用正常值池随机填充完毕，文件：test1_normal_filled.xlsx')