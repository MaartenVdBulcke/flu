import pickle
from pathlib import Path
from pandas import read_csv
from os.path import exists
from sklearn.model_selection import train_test_split

from eval import Evaluator
from models.architectures import SvcFactory
from pipelines.dummy import preprocess
from submission_factory import SubmissionFactory


data_base_path = Path(r'data/')
train_X_path = Path(data_base_path / 'training_set_features.csv')
train_y_path = Path(data_base_path / 'training_set_labels.csv')
test_X_path = Path(data_base_path / 'test_set_features.csv')

model_files_base_path = Path(r'ml_factory/models/.files/')
model_path_h1n1 = Path(model_files_base_path / 'svc_h1n1.pkl')
model_path_seas = Path(model_files_base_path / 'svc_seas.pkl')



if __name__ == '__main__':

    X_train = read_csv(train_X_path)
    y_train = read_csv(train_y_path)

    X, y = preprocess(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=85)



    if not exists(model_path_h1n1):
        print('\n\tTRAINING H1N1 SVC MODEL')
        model_h1n1 = SvcFactory.initialise()
        model_h1n1.fit(X_train, y_train['h1n1_vaccine'])
        with open(model_path_h1n1, 'wb') as f:
            pickle.dump(model_h1n1, f)
    else:
        with open(model_path_h1n1, 'rb') as f:
            model_h1n1 = pickle.load(f)

    if not exists(model_path_seas):
        print('\n\tTRAINING SEASONAL SVC MODEL')
        model_seas = SvcFactory.initialise()
        model_seas.fit(X_train, y_train['seasonal_vaccine'])
        with open(model_path_seas, 'wb') as f:
            pickle.dump(model_seas, f)
    else:
        with open(model_path_seas, 'rb') as f:
            model_seas = pickle.load(f)
    
    y_preds_h1n1_val = model_h1n1.predict(X_val)
    y_preds_seas_val = model_seas.predict(X_val)

    classification_report_h1n1 = Evaluator.get_classification_report(y_val['h1n1_vaccine'], y_preds_h1n1_val)
    roc_auc_score_h1n1 = Evaluator.get_roc_auc_score(y_val['h1n1_vaccine'], y_preds_h1n1_val)

    classification_report_seas = Evaluator.get_classification_report(y_val['seasonal_vaccine'], y_preds_seas_val)
    roc_auc_score_seas = Evaluator.get_roc_auc_score(y_val['seasonal_vaccine'], y_preds_seas_val)

    print(classification_report_h1n1)
    print(roc_auc_score_h1n1)

    print(classification_report_seas)
    print(roc_auc_score_seas)


    # submission
    X_test = read_csv(test_X_path)

    ids =  X_test['respondent_id']
    X_test, _ = preprocess(X_test)

    y_preds_proba_test_h1n1 = model_h1n1.predict_proba(X_test)
    y_preds_proba_test_seas = model_seas.predict_proba(X_test)

    SubmissionFactory.make_submission_file(ids, y_preds_proba_test_h1n1, y_preds_proba_test_seas, title_part='svc_dummy')
