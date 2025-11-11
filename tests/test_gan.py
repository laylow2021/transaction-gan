import pytest

pytest.importorskip("sdv")

from transaction_gan.gan import GANTrainingConfig, SyntheticDataGenerator


def test_gan_generates_samples():
    real_data = [[(i + j) / 10.0 for j in range(4)] for i in range(32)]

    config = GANTrainingConfig(
        epochs=5,
        noise_dim=32,
        hidden_dim=64,
        learning_rate=2e-4,
        batch_size=32,
    )
    generator = SyntheticDataGenerator(len(real_data[0]), config)
    generator.train(real_data)

    samples = generator.generate(5)
    assert len(samples) == 5
    assert len(samples[0]) == len(real_data[0])
