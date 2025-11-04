from transaction_gan.preprocessing import TransactionPreprocessor


def test_preprocessor_round_trip():
    records = [
        {
            "transaction_id": "1",
            "amount": "10.0",
            "merchant_category": "A",
            "transaction_type": "Online",
        },
        {
            "transaction_id": "2",
            "amount": "20.0",
            "merchant_category": "B",
            "transaction_type": "Card",
        },
        {
            "transaction_id": "3",
            "amount": "30.0",
            "merchant_category": "A",
            "transaction_type": "Online",
        },
    ]

    preprocessor = TransactionPreprocessor(
        numeric_features=["amount"],
        categorical_features=["merchant_category", "transaction_type"],
        drop_features=["transaction_id"],
    )

    transformed = preprocessor.fit_transform(records)
    assert len(transformed[0]) == 1 + len({"A", "B", ""}) + len({"Online", "Card", ""})

    reconstructed = preprocessor.inverse_transform(transformed)
    assert {row["merchant_category"] for row in reconstructed} <= {"A", "B", ""}
