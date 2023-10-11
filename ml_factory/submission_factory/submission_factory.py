from pathlib import Path
from pandas import DataFrame
from datetime import datetime 


class SubmissionFactory:

    SUBMISSION_PATH = Path(r'ml_factory\submission_factory\.files')

    @classmethod
    def make_submission_file(cls, ids, preds_h1n1, preds_seas, title_part: str):

        now = datetime.now().strftime('%Y%m%d%H_%M_%S')
        
        df = DataFrame({
            'respondent_id': ids,
            'h1n1_vaccine': preds_h1n1[:, 1],
            'seasonal_vaccine': preds_seas[:, 1]
        })
        
        df.to_csv(Path(cls.SUBMISSION_PATH / f'submission_{title_part}_{now}.csv'), index=False)
