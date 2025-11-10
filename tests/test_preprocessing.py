from transaction_gan.preprocessing import TransactionPreprocessor


def test_preprocessor_round_trip_with_date_and_geo():
    records = [
        {
            "transaction_id": "1",
            "amount": "10.0",
            "merchant_category": "A",
            "transaction_type": "Online",
            "transaction_date": "2024-01-01",
            "transaction_address": "500 Main St, Springfield, IL",
        },
        {
            "transaction_id": "2",
            "amount": "20.0",
            "merchant_category": "B",
            "transaction_type": "Card",
            "transaction_date": "2024-02-15",
            "transaction_address": "33 Oak Ave, Riverton, NY",
        },
    ]

    preprocessor = TransactionPreprocessor(
        continuous_features=["amount"],
        categorical_features=["merchant_category", "transaction_type"],
        date_feature="transaction_date",
        geo_feature="transaction_address",
        id_feature="transaction_id",
        drop_features=["transaction_id"],
    )

    transformed = preprocessor.fit_transform(records)
    # amount + date + (lat, lon) + categorical encodings
    expected_length = (
        1  # amount
        + 1  # date
        + 2  # geo lat/lon
        + len({"A", "B", ""})  # merchant_category
        + len({"Online", "Card", ""})  # transaction_type
    )
    assert len(transformed[0]) == expected_length

    reconstructed = preprocessor.inverse_transform(transformed)
    categories = {row["merchant_category"] for row in reconstructed}
    assert {"A", "B"}.issubset(categories)
    assert all("transaction_address_lat" in row for row in reconstructed)
    assert all(row["transaction_id"].startswith("synthetic_") for row in reconstructed)
