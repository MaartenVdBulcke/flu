from sklearn.svm import SVC 


class SvcFactory:

    @staticmethod
    def initialise(kernel: str = 'rbf', random_state: int = 85) -> SVC:
        return SVC(kernel=kernel, random_state=random_state, probability=True)
