"""合规 MLP 头训练器：≤2 层、≤100K 参数，validation 早停，多种子平均。

- 回归：标签按训练集均值/方差标准化，Huber 损失；预测时反标准化。
- 二分类：BCEWithLogits，输出正类概率。
- 多分类：CrossEntropy，输出 softmax 概率（列序 = 排序后的类别标签）。
多种子平均：回归在原始单位平均；分类在概率空间平均。
"""

import math

import numpy as np
import torch
from torch import nn

PARAM_BUDGET = 100_000


class MLPHead(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out)
        )

    def forward(self, x):
        return self.net(x)


def param_count(d_in, hidden, d_out):
    return d_in * hidden + hidden + hidden * d_out + d_out


def fit_hidden_dim(d_in, d_out, budget=PARAM_BUDGET, cap=96):
    """预算内最大 hidden，上限 cap 防止小任务过宽。"""
    h = min(cap, (budget - d_out) // (d_in + 1 + d_out))
    if h < 4:
        raise ValueError(f"输入维度 {d_in} 在预算 {budget} 下放不下 MLP")
    return int(h)


def _train_single(Xtr, ytr, Xva, yva, task_type, n_class, seed,
                  max_epochs=300, patience=30, lr=1e-3, batch_size=256,
                  weight_decay=1e-4, device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d_in = Xtr.shape[1]
    d_out = 1 if task_type in ("regression", "binary_classification") else n_class
    hidden = fit_hidden_dim(d_in, d_out)
    model = MLPHead(d_in, hidden, d_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task_type == "regression":
        loss_fn = nn.HuberLoss(delta=1.0)
    elif task_type == "binary_classification":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).float().to(device)
    if task_type == "multiclass_classification":
        ytr_t = ytr_t.long()
    Xva_t = torch.from_numpy(Xva).float().to(device)
    yva_t = torch.from_numpy(yva).float().to(device)
    if task_type == "multiclass_classification":
        yva_t = yva_t.long()

    n = Xtr_t.shape[0]
    best_val, best_state, bad = math.inf, None, 0
    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx]).squeeze(-1)
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t).squeeze(-1), yva_t).item()
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def _predict(model, X, task_type, batch_size=4096, device="cpu"):
    outs = []
    X_t = torch.from_numpy(X).float()
    with torch.no_grad():
        for i in range(0, X_t.shape[0], batch_size):
            out = model(X_t[i:i + batch_size].to(device)).squeeze(-1)
            if task_type == "binary_classification":
                out = torch.sigmoid(out)
            elif task_type == "multiclass_classification":
                out = torch.softmax(out, dim=-1)
            outs.append(out.cpu().numpy())
    result = np.concatenate(outs, axis=0)
    return result


def train_and_predict(Xtr, ytr_raw, Xva, Xte, task_type, class_labels=None,
                      seeds=(0, 1, 2, 3), device="cpu"):
    """训练多种子 MLP 并返回 (val_pred, test_pred, meta)。

    回归：ytr_raw 原始单位，内部标准化；val/test 预测已反标准化。
    分类：ytr_raw 为标签字符串/整数；返回概率（二分类 [N]，多分类 [N, C]）。
    """
    Xtr = np.asarray(Xtr, dtype=np.float32)
    Xva = np.asarray(Xva, dtype=np.float32)
    Xte = np.asarray(Xte, dtype=np.float32) if Xte is not None else None

    classes = None
    if task_type == "regression":
        y = np.asarray(ytr_raw, dtype=np.float64)
        y_mean, y_std = float(y.mean()), float(y.std())
        if y_std <= 0:
            raise ValueError("训练标签方差为 0")
        ytr = ((y - y_mean) / y_std).astype(np.float32)
    elif task_type == "binary_classification":
        ytr = np.asarray([int(v) for v in ytr_raw], dtype=np.float32)
    elif task_type in ("multiclass_classification",):
        classes = sorted({str(v) for v in ytr_raw})
        if class_labels is not None:
            classes = [str(c) for c in class_labels]
        lut = {c: i for i, c in enumerate(classes)}
        ytr = np.asarray([lut[str(v)] for v in ytr_raw], dtype=np.int64)
    else:
        raise NotImplementedError(task_type)

    # 从 train 切 10% 作早停验证（不碰官方 validation 做早停，官方 val 只用于评分）
    rng = np.random.default_rng(12345)
    n = Xtr.shape[0]
    perm = rng.permutation(n)
    n_es = max(1, int(0.1 * n))
    es_idx, tr_idx = perm[:n_es], perm[n_es:]

    val_preds, test_preds = [], []
    for seed in seeds:
        model = _train_single(
            Xtr[tr_idx], ytr[tr_idx], Xtr[es_idx], ytr[es_idx],
            task_type, len(classes) if classes else None, seed, device=device,
        )
        vp = _predict(model, Xva, task_type, device=device)
        val_preds.append(vp)
        if Xte is not None:
            test_preds.append(_predict(model, Xte, task_type, device=device))

    val_pred = np.mean(val_preds, axis=0)
    if task_type == "regression":
        val_pred = val_pred * y_std + y_mean
        test_pred = np.mean(test_preds, axis=0) * y_std + y_mean if Xte is not None else None
    else:
        test_pred = np.mean(test_preds, axis=0) if Xte is not None else None

    d_out = 1 if task_type != "multiclass_classification" else len(classes)
    meta = {
        "d_in": int(Xtr.shape[1]),
        "hidden": fit_hidden_dim(Xtr.shape[1], d_out),
        "params_per_head": param_count(Xtr.shape[1], fit_hidden_dim(Xtr.shape[1], d_out), d_out),
        "seeds": list(seeds),
        "classes": classes,
        "n_train": int(n),
    }
    return val_pred, test_pred, meta
