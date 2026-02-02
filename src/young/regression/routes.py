from flask import Blueprint

bp = Blueprint(
    "young_regression",
    __name__,
    url_prefix="/young/regression"
)

@bp.route("/predict")
def predict():
    return "regression dd"