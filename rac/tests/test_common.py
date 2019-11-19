import pytest

from sklearn.utils.estimator_checks import check_estimator

from rac import RacClassifier


@pytest.mark.parametrize(
    "Estimator", [RacClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
