import yaml
import torch
from med_ts_clustering.dataset.data import FeatureConfig
from med_ts_clustering.dataset.utils import prepare_dataset
from med_ts_clustering.models import TimeSeriesClusteringModel
from med_ts_clustering.pretrain_trainer import train_mrm_pretrain
from med_ts_clustering.DEC_trainer import train_dec
from med_ts_clustering.inference import export_clusters
# =====================
# Standalone 训练入口
# =====================
if __name__ == "__main__":

    # 1. 加载 YAML 配置
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg = cfg["config"]
    
    # 2. 构建 feature 配置
    feature_cfg = FeatureConfig(
        patient_id_col=cfg["feature"]["patient_id_col"],
        time_col=cfg["feature"]["time_col"],
        cont_cols=cfg["feature"]["cont_cols"],
        cat_cols=cfg["feature"]["cat_cols"],
        static_cont_cols=cfg["feature"]["static_cont_cols"],
        static_cat_cols=cfg["feature"]["static_cat_cols"],
    )

    # 3. 准备 dataset
    dataset = prepare_dataset(
        cfg["paths"]["static_csv"],
        cfg["paths"]["events_csv"],
        feature_cfg,
        cfg["model"]["max_seq_len"],
    )

    # 4. 模型构建
    n_cont = len(feature_cfg.cont_cols) + len(feature_cfg.static_cont_cols)
    cat_card = []
    for col in feature_cfg.cat_cols:
        cat_card.append(len(dataset.artifacts.cat_vocab_maps[col]))
    for col in feature_cfg.static_cat_cols:
        cat_card.append(len(dataset.artifacts.static_cat_vocab_maps[col]))

    model = TimeSeriesClusteringModel(
        n_cont_features=n_cont,
        cat_cardinalities=cat_card,
        d_model=cfg["model"]["d_model"],
        n_clusters=cfg["model"]["n_clusters"],
        time_transformer_cfg=cfg["model"]["time_transformer_cfg"],
        ft_kwargs=cfg["model"]["ft_kwargs"],
        enable_reconstruction=True,
        use_missing_embedding=True,
        use_mask_embedding=True,
    )

    print("======================= 开始 MRM 预训练 =======================")
    train_mrm_pretrain(model, dataset, cfg)
    print("======================= 预训练完成 =======================")

    # model.load_state_dict(torch.load(cfg["paths"]["model_path"]))
    # print("======================= 开始 DEC 微调 =======================")
    # train_dec(model, dataset, cfg)
    # print("======================= DEC 微调完成 =======================")
    # print("======================= 开始聚类结果导出 =======================")
    # export_clusters(model, dataset, feature_cfg, cfg)
    # print("======================= 聚类结果导出完成 =======================")