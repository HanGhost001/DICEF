"""Class-balanced evaluation and paired bootstrap of aligned predictions."""
import numpy as np
from .mocef import CLASS_NAMES, labels, probabilities


def classification_metrics(y, p):
    p = probabilities(p)
    y = labels(y,len(p))
    cm = np.bincount(3*y+p.argmax(1),minlength=9).reshape(3,3)
    actual,predicted = cm.sum(1),cm.sum(0)
    if (actual == 0).any():
        raise ValueError("Metrics require all three true classes")
    recall = np.diag(cm)/actual
    f1 = np.divide(2*np.diag(cm),actual+predicted,out=np.zeros(3),where=(actual+predicted)>0)
    return {"bacc":float(recall.mean()),"macro_f1":float(f1.mean()),
        "accuracy":float(np.trace(cm)/cm.sum()),
        "recall":dict(zip(CLASS_NAMES,recall.tolist())),"confusion_matrix":cm.tolist()}


def paired_bootstrap(y, a, b, centers=None, samples=20000, seed=20260807):
    a,b = probabilities(a),probabilities(b)
    if a.shape != b.shape or samples < 1:
        raise ValueError("Invalid paired predictions or bootstrap sample count")
    y = labels(y,len(a))
    rng = np.random.default_rng(seed)
    if centers is None:
        pools = [np.flatnonzero(y == c) for c in range(3)]
        draw = lambda:np.concatenate([rng.choice(pool,len(pool),replace=True) for pool in pools])
    else:
        centers = np.asarray(centers)
        if centers.shape != y.shape:
            raise ValueError("Center IDs are not aligned with predictions")
        groups = np.unique(centers)
        pools = {g:np.flatnonzero(centers==g) for g in groups}
        draw = lambda:np.concatenate([pools[g] for g in rng.choice(groups,len(groups),replace=True)])
    names = ("bacc","macro_f1","accuracy")
    ma,mb = classification_metrics(y,a),classification_metrics(y,b)
    values = []
    for _ in range(samples):
        idx = draw()
        if len(np.unique(y[idx])) != 3:
            continue
        xa,xb = classification_metrics(y[idx],a[idx]),classification_metrics(y[idx],b[idx])
        values.append([xa[name]-xb[name] for name in names])
    if not values:
        raise ValueError("No valid bootstrap sample contains all classes")
    quantiles = np.quantile(values,[0.025,0.975],axis=0)
    return {name:{"delta":ma[name]-mb[name],"ci95":quantiles[:,i].tolist(),
                  "valid_samples":len(values)} for i,name in enumerate(names)}
