"""Feature metadata for translating encoded columns into meaningful labels."""
from __future__ import annotations

from dataclasses import dataclass


# Official German Credit (Statlog) codebook labels.  Keeping this mapping here
# ensures no visualization needs to know about the dataset's A-codes.
CATEGORY_LABELS = {
    "Status_of_existing_checking_account": {
        "A11": "< 0 DM", "A12": "0 to < 200 DM", "A13": ">= 200 DM / salary assignment >= 1 year", "A14": "No checking account"},
    "Credit_history": {
        "A30": "No credits / all repaid duly", "A31": "All credits at this bank repaid", "A32": "Existing credits repaid duly", "A33": "Past payment delay", "A34": "Critical account / other existing credits"},
    "Purpose": {
        "A40": "New car", "A41": "Used car", "A42": "Furniture / equipment", "A43": "Radio / television", "A44": "Domestic appliances", "A45": "Repairs", "A46": "Education", "A47": "Vacation", "A48": "Retraining", "A49": "Business", "A410": "Other"},
    "Savings_account_bonds": {
        "A61": "< 100 DM", "A62": "100 to < 500 DM", "A63": "500 to < 1,000 DM", "A64": ">= 1,000 DM", "A65": "Unknown / no savings account"},
    "Present_employment_since": {
        "A71": "Unemployed", "A72": "< 1 year", "A73": "1 to < 4 years", "A74": "4 to < 7 years", "A75": ">= 7 years"},
    "Personal_status_and_sex": {
        "A91": "Male: divorced / separated", "A92": "Female: divorced / married", "A93": "Male: single", "A94": "Male: married / widowed", "A95": "Female: single"},
    "Other_debtors_guarantors": {"A101": "None", "A102": "Co-applicant", "A103": "Guarantor"},
    "Property": {"A121": "Real estate", "A122": "Building society savings / life insurance", "A123": "Car or other property", "A124": "Unknown / no property"},
    "Other_installment_plans": {"A141": "Bank", "A142": "Stores", "A143": "None"},
    "Housing": {"A151": "Rent", "A152": "Own", "A153": "Free"},
    "Job": {"A171": "Unemployed / unskilled, non-resident", "A172": "Unskilled, resident", "A173": "Skilled employee / official", "A174": "Management / self-employed / highly qualified"},
    "Telephone": {"A191": "None", "A192": "Yes, registered in customer's name"},
    "Foreign_worker": {"A201": "Yes", "A202": "No"},
}


@dataclass(frozen=True)
class FeatureMetadata:
    name: str
    kind: str  # "categorical" or "numerical"
    encoded_columns: tuple[str, ...]
    category_labels: dict[str, str]

    def label_for(self, encoded_column: str) -> str:
        return self.category_labels.get(encoded_column, self.name)


def build_feature_metadata(preprocessor, categorical_features, numerical_features):
    """Build metadata once, rather than parsing transformer prefixes in views."""
    encoded_names = list(preprocessor.get_feature_names_out())
    metadata = {}
    for feature in categorical_features:
        prefix = f"cat__{feature}_"
        columns = tuple(name for name in encoded_names if name.startswith(prefix))
        labels = {
            name: CATEGORY_LABELS.get(feature, {}).get(
                name[len(prefix):], name[len(prefix):].replace("_", " ")
            )
            for name in columns
        }
        metadata[feature] = FeatureMetadata(feature, "categorical", columns, labels)
    for feature in numerical_features:
        column = f"num__{feature}"
        metadata[feature] = FeatureMetadata(feature, "numerical", (column,), {})
    return metadata
