import pytest

from sklearn.utils.estimator_checks import check_estimator

from rac import RAClassifier


@pytest.mark.parametrize(
    "Estimator", [RAClassifier()]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
