import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from sklearn.calibration import calibration_curve

class EvaluationService:
    def compute_all_metrics(self, y_true, y_pred, y_prob):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_prob))
        }

    def compute_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(auc)
        }

    def compute_pr_curve(self, y_true, y_prob):
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        auc = average_precision_score(y_true, y_prob)
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
            'auc': float(auc)
        }

    def compute_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        return {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'labels': ['No Claim', 'Claim']
        }

    def compute_threshold_analysis(self, y_true, y_prob, thresholds=None):
        if thresholds is None:
            thresholds = np.arange(0.1, 0.95, 0.05)
        
        results = []
        for t in thresholds:
            preds = (np.array(y_prob) >= t).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            results.append({
                'threshold': float(t),
                'precision': float(precision_score(y_true, preds, zero_division=0)),
                'recall': float(recall_score(y_true, preds, zero_division=0)),
                'f1': float(f1_score(y_true, preds, zero_division=0))
            })
        return results

    def compute_cost_analysis(self, y_true, y_prob, fp_cost, fn_cost):
        thresholds = np.arange(0.01, 1.0, 0.01)
        best_cost = float('inf')
        opt_t = 0.5
        costs = []
        
        for t in thresholds:
            preds = (np.array(y_prob) >= t).astype(int)
            cm = confusion_matrix(y_true, preds)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                cost = fp * fp_cost + fn * fn_cost
                costs.append({'threshold': float(t), 'expected_cost': float(cost)})
                if cost < best_cost:
                    best_cost = cost
                    opt_t = t
                    
        return {'costs': costs, 'optimal_threshold': float(opt_t), 'min_cost': float(best_cost)}

    def compute_calibration_curve(self, y_true, y_prob):
        fraction_positives, mean_predicted = calibration_curve(y_true, y_prob, n_bins=10)
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_prob)
        return {
            'fraction_positives': fraction_positives.tolist(),
            'mean_predicted': mean_predicted.tolist(),
            'brier_score': float(brier)
        }

    def get_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        return report
