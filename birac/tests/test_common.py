import pytest

from sklearn.utils.estimator_checks import check_estimator

from birac import BiRacClassifier


@pytest.mark.parametrize(
    "Estimator", [BiRacClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
