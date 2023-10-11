from typing import Optional, Sequence
from pandas import get_dummies, DataFrame


class Categorical:

    @staticmethod
    def one_hot_encode(df: DataFrame, columns: Optional[Sequence[str]] = None) -> DataFrame:
        if columns is None:
            columns = [
            'hhs_geo_region', 'census_msa', 
            'employment_occupation', 'employment_industry', 
            'employment_status', 'rent_or_own',
            'marital_status', 'race', 'sex'
        ]
        
        return get_dummies(df, columns=columns)
