## 医疗时序行级聚类框架说明（FT-Transformer + Time Transformer + DEC）

本目录提供了一套用于 **医疗时序数据行级事件聚类** 的完整 PyTorch 实现：

- **行级编码器**：`FTTransformer`（来自上级目录的 `ft_transformer.py`）
- **序列编码器**：`TimeTransformer`（对同一病人内部的时间序列做自注意力）
- **聚类模块**：`DEC`（Deep Embedded Clustering）
- **训练流程**：`trainer.py`（无监督训练）  
最终可以得到每条时序事件的 `cluster_id`。

---

## 1. 数据格式与配置

### 1.1 输入数据

- **静态特征表（每位病人一行）**  示例字段（列名）：
  - `subject_id`, `hadm_id`, `stay_id`
  - `gender`
  - `admission_age`, `charlson_comorbidity_index`
  - `height`, `weight`
  - `sofa`, `sapsii`, `oasis`
  - `ventilated`, `total_mv_hours`
  - `crrt_used`, `crrt_hours`
  - `icu_los_hours`
  - `hospital_expire_flag`, `mortality_28d_post_discharge`
  - `icu_first_gcs_total`, `icu_first_gcs_time`, `icu_first_gcs_source`

- **时序事件表（每位病人多行，按时间排序）**  示例字段（列名）：
  - `stay_id`, `hour`, `hour_index`
  - 生命体征与实验室指标：
    - `heart_rate`, `resp_rate`, `temperature`, `sbp`, `dbp`, `map`
    - `fio2`, `ph`, `pco2`, `po2`, `hco3`, `lactate`, `sao2`
    - `base_excess`, `tco2`, `sodium`, `potassium`, `chloride`, `calcium`
    - `hemoglobin`, `hematocrit`
  - 血管活性药物与液体输入输出：
    - `norepinephrine`, `dopamine`, `epinephrine`, `vasopressin`, `phenylephrine`
    - `in_volume_ml_hr`, `crystalloid_in_ml_hr`, `colloid_in_ml_hr`
    - `out_volume_ml_hr`, `urine_output_ml_hr`
  - 派生指标：
    - `pf_ratio`, `anion_gap`

两张表通过 `stay_id` 关联；`hour_index` 字段用于序列排序。

### 1.2 特征配置 `FeatureConfig`


> 注意  
> - 静态表必须包含 `stay_id` 和所有 `static_*` 列。  
> - 时序表必须包含 `stay_id`、`hour_index` 以及所有 `cont_cols` / `cat_cols` 列。  
> - 连续特征会自动做标准化（`StandardScaler`），类别特征会自动映射为整数 id。
> - 目前data_precess不支持处理全列缺失的列数据，全列为Nan的视为无效数据，不导入featureConfig中

---

## 2. 目录结构概览

- `models.py`  
  - `RowEncoderFTTransformer`：将 `(B, T, *)` 行级特征包装为 `FTTransformer` 输入并输出 `(B, T, d_model)`。  
  - `TimeSeriesClusteringModel`：串联 **FT-Transformer 行编码器 + Time Transformer 序列编码器 + DEC 聚类层**。

- `trainer.py`  
  - `train_dec(...)`：完整的无监督训练流程（包含 KMeans 初始化 + DEC KL 损失训练）。  
  - `export_cluster_assignments(...)`：将最终每条时序事件的 `cluster_id` 导出为 CSV。  
  - 依赖 `dataset` 子包（`prepare_dataset`）完成数据加载与预处理。

- `dataset/`（根据你的实现可能包含以下组件）  
  - `utils.py`：`FeatureConfig` 定义、`prepare_dataset` 工具函数（构造 `Dataset`）。  
  - 其他数据预处理相关模块（构建 `(B, T, feature)`、mask、time 列等）。

- `time_transformer.py`  
  - `TimeTransformer` 与时间/位置编码（按时间步编码序列）。

- `dec.py`  
  - `ClusteringLayer`、`target_distribution`、`dec_loss` 等 DEC 相关组件。

---

## 3. 快速开始训练

### 3.1 安装依赖

确保已安装 PyTorch 以及用于聚类与预处理的依赖：

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn tqdm
```

---

## 4. 训练过程细节（简要）

- **数据预处理**（由 `dataset` 子包完成）：
  - 按 `patient_id` 将静态表与时序表关联。
  - 按 `time` 对时序事件排序。
  - 连续特征做标准化；类别特征做整数编码。
  - 将每位病人的序列截断/填充到 `max_seq_len`，生成 `(B, T, feature)` 与 `mask`。

- **编码与聚类**（由 `TimeSeriesClusteringModel` + `DEC` 完成）：
  - 每一行事件经 **FT-Transformer 行级编码器** 得到 `z_t`。
  - 整个序列经 **Time Transformer** 得到带上下文的 `h_t`。
  - 所有 `h_t` 通过 **DEC 聚类层** 得到 soft assignment `q`；  
    根据 `q` 构造目标分布 `p`，最小化 `KL(p || q)` 训练 encoder 与聚类中心。

---

