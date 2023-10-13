from sklearn.metrics import classification_report, roc_auc_score


class Evaluator:

    @staticmethod
    def get_classification_report(y_true, y_preds):
        return classification_report(y_true, y_preds)

    @staticmethod
    def get_roc_auc_score(y_true, y_preds) -> float:
        return roc_auc_score(y_true, y_preds, average='macro')
