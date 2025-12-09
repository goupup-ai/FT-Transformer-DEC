from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureConfig:
    """Configuration of static & dynamic features."""

    patient_id_col: str = "patient_id"
    time_col: str = "time"

    # dynamic (row-level) features
    cont_cols: Sequence[str] = ()
    cat_cols: Sequence[str] = ()

    # static (per-patient) features (will be broadcast to all time steps)
    static_cont_cols: Sequence[str] = ()
    static_cat_cols: Sequence[str] = ()


@dataclass
class PreprocessArtifacts:
    """Artifacts needed for inference / later use."""

    cont_scaler: Optional[StandardScaler]
    static_cont_scaler: Optional[StandardScaler]
    cat_vocab_maps: Dict[str, Dict[str, int]]
    static_cat_vocab_maps: Dict[str, Dict[str, int]]


class MedicalTimeSeriesDataset(Dataset):
    """Medical time-series dataset for row-level clustering.

    Output for each item:
        - x_cont: FloatTensor[B, T, n_cont_total]
        - x_cat: LongTensor[B, T, n_cat_total]
        - mask: BoolTensor[B, T] (True for valid positions)
        - patient_ids: list of patient ids in the batch (here length == 1; batching is done by DataLoader)
        - times: list of np.ndarray of raw time values for each patient sequence
    """

    def __init__(
        self,
        static_df: pd.DataFrame,
        events_df: pd.DataFrame,
        feature_cfg: FeatureConfig,
        max_seq_len: int,
        artifacts: Optional[PreprocessArtifacts] = None,
        fit_preprocessors: bool = True,
    ) -> None:
        self.feature_cfg = feature_cfg
        self.max_seq_len = max_seq_len

        pid_col = feature_cfg.patient_id_col
        time_col = feature_cfg.time_col

        # sort temporal events
        events_df = events_df.sort_values([pid_col, time_col])

        # keep only configured columns
        needed_event_cols = (
            [pid_col, time_col]
            + list(feature_cfg.cont_cols)
            + list(feature_cfg.cat_cols)
        )
        events_df = events_df[needed_event_cols].copy()

        needed_static_cols = (
            [pid_col]
            + list(feature_cfg.static_cont_cols)
            + list(feature_cfg.static_cat_cols)
        )
        static_df = static_df[needed_static_cols].copy()

        # Record missing values BEFORE preprocessing (for missing_mask)
        # We need to do this before filling/transforming
        self._raw_events_df = events_df.copy()
        self._raw_static_df = static_df.copy()
        
        # preprocessors
        if artifacts is None:
            artifacts = self._fit_preprocessors(static_df, events_df, feature_cfg)
        elif fit_preprocessors:
            # overrides given artifacts if requested
            artifacts = self._fit_preprocessors(static_df, events_df, feature_cfg)
        self.artifacts = artifacts

        # apply preprocessing (this will fill missing values)
        events_df = self._transform_events(events_df)
        static_df = self._transform_static(static_df)

        # build per-patient sequences
        self.sequences = self._build_sequences(static_df, events_df)

    # ------------------------------------------------------------------ #
    # preprocessing helpers
    # ------------------------------------------------------------------ #
    def _fit_preprocessors(
        self,
        static_df: pd.DataFrame,
        events_df: pd.DataFrame,
        cfg: FeatureConfig,
    ) -> PreprocessArtifacts:
        cont_scaler = None
        static_cont_scaler = None
        if cfg.cont_cols:
            cont_scaler = StandardScaler()
            cont_scaler.fit(events_df[list(cfg.cont_cols)].astype(float))
        if cfg.static_cont_cols:
            static_cont_scaler = StandardScaler()
            static_cont_scaler.fit(static_df[list(cfg.static_cont_cols)].astype(float))

        def build_vocab(
            series: pd.Series,
        ) -> Dict[str, int]:
            unique = series.astype(str).fillna("___NA___").unique()
            return {v: i for i, v in enumerate(unique)}

        cat_vocab_maps: Dict[str, Dict[str, int]] = {}
        for col in cfg.cat_cols:
            cat_vocab_maps[col] = build_vocab(events_df[col])

        static_cat_vocab_maps: Dict[str, Dict[str, int]] = {}
        for col in cfg.static_cat_cols:
            static_cat_vocab_maps[col] = build_vocab(static_df[col])

        return PreprocessArtifacts(
            cont_scaler=cont_scaler,
            static_cont_scaler=static_cont_scaler,
            cat_vocab_maps=cat_vocab_maps,
            static_cat_vocab_maps=static_cat_vocab_maps,
        )

    def _transform_events(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.feature_cfg
        art = self.artifacts
        assert art is not None

        # Fill missing continuous values with median (or 0 if all NaN) before scaling
        if cfg.cont_cols:
            cont_df = df[list(cfg.cont_cols)].astype(float)
            # Fill NaN with median per column, or 0 if all NaN
            for col in cfg.cont_cols:
                median_val = cont_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                cont_df[col] = cont_df[col].fillna(median_val)
            df[list(cfg.cont_cols)] = cont_df
            
            if art.cont_scaler is not None:
                df[list(cfg.cont_cols)] = art.cont_scaler.transform(
                    df[list(cfg.cont_cols)]
                )

        for col in cfg.cat_cols:
            vocab = art.cat_vocab_maps[col]
            df[col] = (
                df[col]
                .astype(str)
                .fillna("___NA___")
                .map(lambda v: vocab.get(v, 0))
                .astype(int)
            )

        return df

    def _transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.feature_cfg
        art = self.artifacts
        assert art is not None

        # Fill missing continuous values with median (or 0 if all NaN) before scaling
        if cfg.static_cont_cols:
            static_cont_df = df[list(cfg.static_cont_cols)].astype(float)
            # Fill NaN with median per column, or 0 if all NaN
            for col in cfg.static_cont_cols:
                median_val = static_cont_df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                static_cont_df[col] = static_cont_df[col].fillna(median_val)
            df[list(cfg.static_cont_cols)] = static_cont_df
            
            if art.static_cont_scaler is not None:
                df[list(cfg.static_cont_cols)] = art.static_cont_scaler.transform(
                    df[list(cfg.static_cont_cols)]
                )

        for col in cfg.static_cat_cols:
            vocab = art.static_cat_vocab_maps[col]
            df[col] = (
                df[col]
                .astype(str)
                .fillna("___NA___")
                .map(lambda v: vocab.get(v, 0))
                .astype(int)
            )

        return df

    def _build_sequences(
        self,
        static_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> List[Dict[str, object]]:
        cfg = self.feature_cfg
        pid_col = cfg.patient_id_col
        time_col = cfg.time_col

        static_indexed = static_df.set_index(pid_col)

        # Build index for raw data for efficient lookup
        raw_events_indexed = self._raw_events_df.set_index([pid_col, time_col])
        raw_static_indexed = self._raw_static_df.set_index(pid_col)
        
        sequences: List[Dict[str, object]] = []
        for pid, group in events_df.groupby(pid_col):
            group = group.sort_values(time_col)
            times = group[time_col].to_numpy()

            # Get continuous features and missing masks
            if cfg.cont_cols:
                # Get from preprocessed (already filled) data
                cont_dyn = group[list(cfg.cont_cols)].to_numpy(dtype=np.float32)
                # Get missing mask from raw data before preprocessing
                missing_cont_dyn = np.zeros((len(group), len(cfg.cont_cols)), dtype=bool)
                for idx, (_, row) in enumerate(group.iterrows()):
                    time_val = row[time_col]
                    try:
                        raw_row = raw_events_indexed.loc[(pid, time_val)]
                        if isinstance(raw_row, pd.Series):
                            missing_cont_dyn[idx] = pd.isna(raw_row[list(cfg.cont_cols)]).to_numpy(dtype=bool)
                        else:
                            # Multiple rows with same pid+time, take first
                            missing_cont_dyn[idx] = pd.isna(raw_row.iloc[0][list(cfg.cont_cols)]).to_numpy(dtype=bool)
                    except (KeyError, IndexError):
                        # If not found in raw data, assume no missing
                        pass
            else:
                cont_dyn = np.zeros((len(group), 0), dtype=np.float32)
                missing_cont_dyn = np.zeros((len(group), 0), dtype=bool)

            # Get categorical features and missing masks
            if cfg.cat_cols:
                cat_dyn = group[list(cfg.cat_cols)].to_numpy(dtype=np.int64)
                # Get missing mask from raw data (before preprocessing)
                missing_cat_dyn = np.zeros((len(group), len(cfg.cat_cols)), dtype=bool)
                for idx, (_, row) in enumerate(group.iterrows()):
                    time_val = row[time_col]
                    try:
                        raw_row = raw_events_indexed.loc[(pid, time_val)]
                        if isinstance(raw_row, pd.Series):
                            missing_cat_dyn[idx] = pd.isna(raw_row[list(cfg.cat_cols)]).to_numpy(dtype=bool)
                        else:
                            missing_cat_dyn[idx] = pd.isna(raw_row.iloc[0][list(cfg.cat_cols)]).to_numpy(dtype=bool)
                    except (KeyError, IndexError):
                        pass
            else:
                cat_dyn = np.zeros((len(group), 0), dtype=np.int64)
                missing_cat_dyn = np.zeros((len(group), 0), dtype=bool)

            static_row = static_indexed.loc[pid]
            if cfg.static_cont_cols:
                static_cont = static_row[list(cfg.static_cont_cols)].to_numpy(dtype=np.float32)
                # Get missing mask from raw static data
                try:
                    raw_static_row = raw_static_indexed.loc[pid]
                    if isinstance(raw_static_row, pd.Series):
                        missing_static_cont = pd.isna(raw_static_row[list(cfg.static_cont_cols)]).to_numpy(dtype=bool)
                    else:
                        missing_static_cont = pd.isna(raw_static_row.iloc[0][list(cfg.static_cont_cols)]).to_numpy(dtype=bool)
                except (KeyError, IndexError):
                    missing_static_cont = np.zeros((len(cfg.static_cont_cols),), dtype=bool)
            else:
                static_cont = np.zeros((0,), dtype=np.float32)
                missing_static_cont = np.zeros((0,), dtype=bool)
            
            if cfg.static_cat_cols:
                static_cat = static_row[list(cfg.static_cat_cols)].to_numpy(dtype=np.int64)
                # Get missing mask from raw static data
                try:
                    raw_static_row = raw_static_indexed.loc[pid]
                    if isinstance(raw_static_row, pd.Series):
                        missing_static_cat = pd.isna(raw_static_row[list(cfg.static_cat_cols)]).to_numpy(dtype=bool)
                    else:
                        missing_static_cat = pd.isna(raw_static_row.iloc[0][list(cfg.static_cat_cols)]).to_numpy(dtype=bool)
                except (KeyError, IndexError):
                    missing_static_cat = np.zeros((len(cfg.static_cat_cols),), dtype=bool)
            else:
                static_cat = np.zeros((0,), dtype=np.int64)
                missing_static_cat = np.zeros((0,), dtype=bool)

            # broadcast static features over time
            if static_cont.size > 0:
                static_cont_seq = np.repeat(
                    static_cont[None, :], repeats=len(group), axis=0
                )
                cont_all = np.concatenate([cont_dyn, static_cont_seq], axis=1)
                missing_static_cont_seq = np.repeat(
                    missing_static_cont[None, :], repeats=len(group), axis=0
                )
                missing_cont_all = np.concatenate([missing_cont_dyn, missing_static_cont_seq], axis=1)
            else:
                cont_all = cont_dyn
                missing_cont_all = missing_cont_dyn

            if static_cat.size > 0:
                static_cat_seq = np.repeat(
                    static_cat[None, :], repeats=len(group), axis=0
                )
                cat_all = np.concatenate([cat_dyn, static_cat_seq], axis=1)
                missing_static_cat_seq = np.repeat(
                    missing_static_cat[None, :], repeats=len(group), axis=0
                )
                missing_cat_all = np.concatenate([missing_cat_dyn, missing_static_cat_seq], axis=1)
            else:
                cat_all = cat_dyn
                missing_cat_all = missing_cat_dyn

            sequences.append(
                {
                    "patient_id": pid,
                    "times": times,
                    "cont": cont_all,
                    "cat": cat_all,
                    "missing_cont": missing_cont_all,
                    "missing_cat": missing_cat_all,
                }
            )

        return sequences

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        item = self.sequences[idx]
        cont = item["cont"]  # (T, Cc)
        cat = item["cat"]  # (T, Ck)
        times = item["times"]
        pid = item["patient_id"]
        missing_cont = item.get("missing_cont", np.zeros_like(cont, dtype=bool))  # (T, Cc)
        missing_cat = item.get("missing_cat", np.zeros_like(cat, dtype=bool))  # (T, Ck)

        T = min(len(times), self.max_seq_len)
        # truncate from the end if longer than max_seq_len
        cont = cont[-T:]
        cat = cat[-T:]
        times = times[-T:]
        missing_cont = missing_cont[-T:]
        missing_cat = missing_cat[-T:]

        n_cont = cont.shape[1]
        n_cat = cat.shape[1]

        cont_padded = np.zeros((self.max_seq_len, n_cont), dtype=np.float32)
        cat_padded = np.zeros((self.max_seq_len, n_cat), dtype=np.int64)
        mask = np.zeros((self.max_seq_len,), dtype=bool)
        missing_cont_padded = np.zeros((self.max_seq_len, n_cont), dtype=bool)
        missing_cat_padded = np.zeros((self.max_seq_len, n_cat), dtype=bool)

        cont_padded[:T] = cont
        cat_padded[:T] = cat
        mask[:T] = True
        missing_cont_padded[:T] = missing_cont
        missing_cat_padded[:T] = missing_cat

        return {
            "patient_id": pid,
            "times": times,
            "x_cont": torch.from_numpy(cont_padded),  # (T, Cc)
            "x_cat": torch.from_numpy(cat_padded),  # (T, Ck)
            "mask": torch.from_numpy(mask),  # (T,)
            "missing_cont": torch.from_numpy(missing_cont_padded),  # (T, Cc)
            "missing_cat": torch.from_numpy(missing_cat_padded),  # (T, Ck)
        }


def collate_patient_sequences(batch):
    """Simple collate function: put patients along batch dimension."""
    # all have same max_seq_len and feature sizes by construction
    x_cont = torch.stack([b["x_cont"] for b in batch], dim=0)  # (B, T, Cc)
    x_cat = torch.stack([b["x_cat"] for b in batch], dim=0)  # (B, T, Ck)
    mask = torch.stack([b["mask"] for b in batch], dim=0)  # (B, T)
    patient_ids = [b["patient_id"] for b in batch]
    times = [b["times"] for b in batch]
    
    # Missing masks (optional, may not be present in all batches)
    missing_cont = None
    missing_cat = None
    if "missing_cont" in batch[0]:
        missing_cont = torch.stack([b["missing_cont"] for b in batch], dim=0)  # (B, T, Cc)
    if "missing_cat" in batch[0]:
        missing_cat = torch.stack([b["missing_cat"] for b in batch], dim=0)  # (B, T, Ck)
    
    result = {
        "patient_ids": patient_ids,
        "times": times,
        "x_cont": x_cont,
        "x_cat": x_cat,
        "mask": mask,
    }
    if missing_cont is not None:
        result["missing_cont"] = missing_cont
    if missing_cat is not None:
        result["missing_cat"] = missing_cat
    
    return result



