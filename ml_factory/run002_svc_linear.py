import pickle
from pathlib import Path
from numpy import mean
from pandas import read_csv
from os.path import exists
from sklearn.model_selection import train_test_split

from pipelines.dummy import preprocess

from eval import Evaluator
from models.architectures import SvcFactory
from feature_importance import FeatureImportance
from submission_factory import SubmissionFactory

data_base_path = Path(r'data/')
train_X_path = Path(data_base_path / 'training_set_features.csv')
train_y_path = Path(data_base_path / 'training_set_labels.csv')
test_X_path = Path(data_base_path / 'test_set_features.csv')

model_files_base_path = Path(r'ml_factory/models/.files/')
model_path_h1n1 = Path(model_files_base_path / 'svc_lineaer_h1n1.pkl')
model_path_seas = Path(model_files_base_path / 'svc_linear_seas.pkl')


if __name__ == '__main__':

    X_train = read_csv(train_X_path)
    y_train = read_csv(train_y_path)

    X, y = preprocess(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=85)

    if not exists(model_path_h1n1):
        print('\n\tTRAINING H1N1 SVC MODEL')
        model_h1n1 = SvcFactory.initialise(kernel='linear')
        model_h1n1.fit(X_train, y_train['h1n1_vaccine'])
        with open(model_path_h1n1, 'wb') as f:
            pickle.dump(model_h1n1, f)
    else:
        with open(model_path_h1n1, 'rb') as f:
            model_h1n1 = pickle.load(f)

    if not exists(model_path_seas):
        print('\n\tTRAINING SEASONAL SVC MODEL')
        model_seas = SvcFactory.initialise(kernel='linear')
        model_seas.fit(X_train, y_train['seasonal_vaccine'])
        with open(model_path_seas, 'wb') as f:
            pickle.dump(model_seas, f)
    else:
        with open(model_path_seas, 'rb') as f:
            model_seas = pickle.load(f)
    
    # eval
    y_preds_proba_h1n1 = model_h1n1.predict_proba(X_val)
    y_preds_proba_seas = model_seas.predict_proba(X_val)
    
    roc_val_h1n1 = Evaluator.get_roc_auc_score(y_val['h1n1_vaccine'], y_preds_proba_h1n1[:, 1])
    roc_val_seas = Evaluator.get_roc_auc_score(y_val['seasonal_vaccine'], y_preds_proba_seas[:, 1])
    roc_mean = mean((roc_val_h1n1, roc_val_seas))
    print(
        f'VALIDATION ROC_AUC_SCORES:\n\th1n1: {round(roc_val_h1n1, 2)}\n\tseas: {round(roc_val_seas, 2)}\n\tmean: {round(roc_mean, 2)}'
    )

    # submission
    X_test = read_csv(test_X_path)

    ids =  X_test['respondent_id']
    X_test, _ = preprocess(X_test)

    y_preds_proba_test_h1n1 = model_h1n1.predict_proba(X_test)
    y_preds_proba_test_seas = model_seas.predict_proba(X_test)

    SubmissionFactory.make_submission_file(ids, y_preds_proba_test_h1n1, y_preds_proba_test_seas, title_part='svc_linear')

    FeatureImportance.save_feature_importance_plot(model_h1n1, X_train.columns, 'svc_linear_h1n1')
    FeatureImportance.save_feature_importance_plot(model_seas, X_train.columns, 'svc_linear_seas')
