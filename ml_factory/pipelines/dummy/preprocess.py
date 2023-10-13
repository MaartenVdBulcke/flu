from numpy import array
from typing import Optional, Sequence
from pandas import merge, DataFrame

from pipelines.common import Ordinal, Categorical


def preprocess(X: DataFrame, y: Optional[DataFrame] = None) -> Sequence[array]:
    
    X = Categorical.one_hot_encode(X, columns=None)
    X['age_group'] = X.age_group.apply(lambda _: Ordinal.age_group(_))
    X['education'] = X.education.apply(lambda _: Ordinal.education(_))
    X['income_poverty'] = X.income_poverty.apply(lambda _: Ordinal.ordinal_poverty(_))

    if y is not None:
        df = merge(X, y, on='respondent_id')
        df.dropna(inplace=True)
        X = df.drop(['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine'], axis=1)
        y = df[['h1n1_vaccine', 'seasonal_vaccine']]
    else:
        X.fillna(-1, inplace=True)
        X = X.drop(['respondent_id'], axis=1)

    return X, y
