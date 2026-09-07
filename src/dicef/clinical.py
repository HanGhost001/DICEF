"""Training-only class-balanced multinomial logistic regression."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import CLINICAL_NAMES
from .mocef import CLASS_NAMES, labels, softmax

DEFAULT_C_GRID = (0.01,0.03,0.1,0.3,1,3,10)


def fit_pipeline(x,y,c):
    if np.isnan(x).all(axis=0).any():
        raise ValueError("A clinical feature is entirely missing in a fitting partition")
    model = make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
        LogisticRegression(C=float(c),class_weight="balanced",max_iter=5000,solver="lbfgs"))
    with warnings.catch_warnings():
        warnings.simplefilter("error",ConvergenceWarning)
        model.fit(x,y)
    return model


class ClinicalClassifier:
    def __init__(self):
        self.state = None

    def fit(self, train_x, train_y, c_grid=DEFAULT_C_GRID, seed=42):
        x = np.asarray(train_x,dtype=np.float64)
        y = labels(train_y,len(x))
        if x.ndim != 2 or x.shape[1] != 17 or np.isinf(x).any():
            raise ValueError("Expected 17 clinical features; NaN is allowed, infinity is not")
        if (np.bincount(y,minlength=3) < 3).any():
            raise ValueError("Clinical three-fold selection requires at least three training samples per class")
        grid = sorted(set(float(c) for c in c_grid))
        if not grid or any(not np.isfinite(c) or c <= 0 for c in grid):
            raise ValueError("C candidates must be finite and positive")
        cv = list(StratifiedKFold(3,shuffle=True,random_state=seed).split(x,y))
        best_score, best_c, scores = -np.inf, None, []
        for c in grid:
            score = float(np.mean([balanced_accuracy_score(y[val],
                fit_pipeline(x[fit],y[fit],c).predict(x[val])) for fit,val in cv]))
            scores.append({"C":c,"mean_bacc":score})
            if score > best_score:
                best_score,best_c = score,c
        model = fit_pipeline(x,y,best_c)
        imputer,scaler,lr = model.steps[0][1],model.steps[1][1],model.steps[2][1]
        if list(lr.classes_) != [0,1,2]:
            raise ValueError("Unexpected class order")
        # Store only numeric parameters, not a pickle executable on loading.
        self.state = {"classes":list(CLASS_NAMES),"features":list(CLINICAL_NAMES),
            "median":imputer.statistics_.tolist(),"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),
            "coef":lr.coef_.tolist(),"intercept":lr.intercept_.tolist(),
            "selected_C":best_c,"cv_seed":seed,"cv_scores":scores}
        np.testing.assert_allclose(self.predict_proba(x),model.predict_proba(x),rtol=1e-12,atol=1e-12)
        return self

    def predict_proba(self,x):
        if self.state is None:
            raise RuntimeError("Fit or load the clinical classifier first")
        x = np.asarray(x,dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 17 or not len(x) or np.isinf(x).any():
            raise ValueError("Invalid clinical feature matrix")
        x = np.where(np.isnan(x),np.asarray(self.state["median"]),x)
        x = (x-np.asarray(self.state["mean"]))/np.asarray(self.state["scale"])
        return softmax(x @ np.asarray(self.state["coef"]).T + np.asarray(self.state["intercept"]))

    @classmethod
    def from_dict(cls,state):
        if state.get("classes") != list(CLASS_NAMES) or state.get("features") != list(CLINICAL_NAMES):
            raise ValueError("Class or feature order differs from the DICEF contract")
        for key,shape in (("median",(17,)),("mean",(17,)),("scale",(17,)),("coef",(3,17)),("intercept",(3,))):
            array = np.asarray(state[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid clinical parameter: {key}")
        if (np.asarray(state["scale"]) <= 0).any():
            raise ValueError("Clinical scales must be positive")
        model = cls()
        model.state = state
        return model
