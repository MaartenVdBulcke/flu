from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureImportance:

    @staticmethod
    def save_feature_importance_plot(model, cols, title: str, base_path= Path('.plots')):

        plt.figure(figsize=(8,16))
        sns.barplot(y=cols, x=model.coef_[0], orient='h')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(Path(base_path / f'feature_importance_{title}.jpg'))
        plt.close()
