"""评分器：对齐赛题指标的本地实现。

- 回归: score = clip(1 - MAE / MAE_baseline, 0, 1)，baseline = 常数 train_median 模型，
  MAE 按观测计（同一样本多次测量各自计入、共用该样本的预测值）。
- 二分类: ROC-AUC。
- 多分类: macro one-vs-one ROC-AUC。
- 逐残基分类: 宏平均 OVO AUC，按残基展平（P4 实现）。
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def score_regression(obs_by_id, pred_by_id, train_median):
    """obs_by_id: {id: [观测值...]}; pred_by_id: {id: 预测值}。返回 (score, mae, baseline_mae)。"""
    errs, base_errs = [], []
    missing = 0
    for rid, obs in obs_by_id.items():
        if rid not in pred_by_id:
            missing += 1
            continue
        p = float(pred_by_id[rid])
        for v in obs:
            errs.append(abs(p - v))
            base_errs.append(abs(train_median - v))
    if not errs:
        raise ValueError("没有任何可评分的观测")
    if missing:
        raise ValueError(f"有 {missing} 条记录缺预测，赛规按缺预测处理（应先补全再评分）")
    mae = float(np.mean(errs))
    baseline = float(np.mean(base_errs))
    if baseline <= 0:
        raise ValueError("基线 MAE 非正，无法归一化")
    score = min(max(1.0 - mae / baseline, 0.0), 1.0)
    return score, mae, baseline


def score_binary(y_true, y_score):
    """y_true: 0/1 列表（容忍 '0.0'/'1.0' 字符串）; y_score: 正类概率。返回 ROC-AUC。"""
    y_true = np.asarray([int(float(v)) for v in y_true])
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(set(y_true.tolist())) < 2:
        raise ValueError("验证集只有一个类别，AUC 未定义")
    return float(roc_auc_score(y_true, y_score))


def score_multiclass(y_true, y_prob, class_labels):
    """y_prob: [N, C] 概率，列顺序对应 class_labels。返回 macro OVO AUC。"""
    y_true = [str(v) for v in y_true]
    classes = [str(c) for c in class_labels]
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.shape[1] != len(classes):
        raise ValueError("概率列数与类别数不一致")
    return float(roc_auc_score(
        y_true, y_prob, multi_class="ovo", average="macro", labels=classes
    ))
