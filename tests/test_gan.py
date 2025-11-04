from transaction_gan.gan import GANTrainingConfig, SyntheticDataGenerator


def test_gan_generates_samples():
    real_data = [[(i + j) / 10.0 for j in range(4)] for i in range(32)]

    config = GANTrainingConfig(epochs=50, noise_dim=5, hidden_dim=8, learning_rate=5e-3)
    generator = SyntheticDataGenerator(len(real_data[0]), config)
    generator.train(real_data)

    samples = generator.generate(5)
    assert len(samples) == 5
    assert len(samples[0]) == len(real_data[0])
