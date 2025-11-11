from transaction_gan.preprocessing import TransactionPreprocessor


def test_preprocessor_round_trip_with_string_columns():
    records = [
        {
            "transaction_id": "1",
            "amount": "10.0",
            "merchant_category": "A",
            "transaction_type": "Online",
            "transaction_date": "2024-01-01",
            "transaction_address": "Springfield",
        },
        {
            "transaction_id": "2",
            "amount": "20.0",
            "merchant_category": "B",
            "transaction_type": "Card",
            "transaction_date": "2024-02-15",
            "transaction_address": "Riverton",
        },
    ]

    preprocessor = TransactionPreprocessor(
        continuous_features=["amount"],
        categorical_features=[
            "merchant_category",
            "transaction_type",
            "transaction_date",
            "transaction_address",
        ],
        id_feature="transaction_id",
        drop_features=["transaction_id"],
    )

    transformed = preprocessor.fit_transform(records)
    expected_length = (
        1  # amount
        + len({"A", "B", ""})
        + len({"Online", "Card", ""})
        + len({"2024-01-01", "2024-02-15", ""})
        + len({"Springfield", "Riverton", ""})
    )
    assert len(transformed[0]) == expected_length

    reconstructed = preprocessor.inverse_transform(transformed)
    categories = {row["merchant_category"] for row in reconstructed}
    assert {"A", "B"}.issubset(categories)
    assert all("transaction_address" in row for row in reconstructed)
    assert all(row["transaction_id"].startswith("synthetic_") for row in reconstructed)


def test_preprocessor_preserves_hierarchical_relationships():
    records = [
        {"id": "1", "category": "Grocery", "subcategory": "Produce"},
        {"id": "2", "category": "Grocery", "subcategory": "Bakery"},
        {"id": "3", "category": "Electronics", "subcategory": "Phones"},
    ]

    preprocessor = TransactionPreprocessor(
        continuous_features=[],
        categorical_features=["category", "subcategory"],
        hierarchical_categorical_groups=[("category", "subcategory")],
        id_feature="id",
    )

    transformed = preprocessor.fit_transform(records)
    # 3 observed combos + fallback bucket
    assert preprocessor.output_dim == 4
    reconstructed = preprocessor.inverse_transform(transformed)
    combos = {(row["category"], row["subcategory"]) for row in reconstructed}
    assert combos == {
        ("Grocery", "Produce"),
        ("Grocery", "Bakery"),
        ("Electronics", "Phones"),
    }
