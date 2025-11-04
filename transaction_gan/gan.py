"""Pure Python implementation of a simple GAN for transaction data."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


def random_matrix(rows: int, cols: int, scale: float) -> Matrix:
    return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def random_vector(length: int, scale: float) -> Vector:
    return [random.uniform(-scale, scale) for _ in range(length)]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    th = math.tanh(x)
    return 1.0 - th * th


def matvec_mul(matrix: Matrix, vector: Vector) -> Vector:
    return [sum(m_ij * v_j for m_ij, v_j in zip(row, vector)) for row in matrix]



def vector_add(vec: Vector, bias: Vector) -> Vector:
    return [x + b for x, b in zip(vec, bias)]




def zeros_like(matrix: Matrix) -> Matrix:
    return [[0.0 for _ in row] for row in matrix]


def zeros_vector(length: int) -> Vector:
    return [0.0 for _ in range(length)]


@dataclass
class LayerParameters:
    w1: Matrix
    b1: Vector
    w2: Matrix
    b2: Vector


@dataclass
class LayerGradients:
    w1: Matrix
    b1: Vector
    w2: Matrix
    b2: Vector


@dataclass
class DiscriminatorCache:
    input_vector: Vector
    hidden_linear: Vector
    hidden_activation: Vector
    output_linear: float
    output_activation: float


@dataclass
class GeneratorCache:
    noise: Vector
    hidden_linear: Vector
    hidden_activation: Vector


@dataclass
class GANTrainingConfig:
    noise_dim: int = 8
    hidden_dim: int = 16
    epochs: int = 200
    learning_rate: float = 1e-3


class TransactionGenerator:
    """A lightweight generator network for tabular data."""

    def __init__(self, noise_dim: int, output_dim: int, hidden_dim: int) -> None:
        scale = 1.0 / math.sqrt(noise_dim)
        self.params = LayerParameters(
            w1=random_matrix(hidden_dim, noise_dim, scale),
            b1=random_vector(hidden_dim, scale),
            w2=random_matrix(output_dim, hidden_dim, scale),
            b2=random_vector(output_dim, scale),
        )

    def forward(self, noise: Vector) -> Tuple[Vector, GeneratorCache]:
        z1 = vector_add(matvec_mul(self.params.w1, noise), self.params.b1)
        h1 = [tanh(value) for value in z1]
        output = vector_add(matvec_mul(self.params.w2, h1), self.params.b2)
        cache = GeneratorCache(noise=noise, hidden_linear=z1, hidden_activation=h1)
        return output, cache

    def backward(self, cache: GeneratorCache, grad_output: Vector) -> LayerGradients:
        grad_w2 = zeros_like(self.params.w2)
        grad_b2 = zeros_vector(len(self.params.b2))
        grad_hidden = zeros_vector(len(cache.hidden_activation))

        for i, grad in enumerate(grad_output):
            grad_b2[i] = grad
            for j, activation in enumerate(cache.hidden_activation):
                grad_w2[i][j] += grad * activation
                grad_hidden[j] += grad * self.params.w2[i][j]

        grad_hidden_linear = [grad_hidden[i] * tanh_derivative(cache.hidden_linear[i]) for i in range(len(grad_hidden))]

        grad_w1 = zeros_like(self.params.w1)
        grad_b1 = zeros_vector(len(self.params.b1))
        for i, grad in enumerate(grad_hidden_linear):
            grad_b1[i] = grad
            for j, noise_value in enumerate(cache.noise):
                grad_w1[i][j] += grad * noise_value

        return LayerGradients(w1=grad_w1, b1=grad_b1, w2=grad_w2, b2=grad_b2)

    def apply_gradients(self, grads: LayerGradients, learning_rate: float) -> None:
        for i, row in enumerate(self.params.w1):
            for j, _ in enumerate(row):
                row[j] -= learning_rate * grads.w1[i][j]
        for i, value in enumerate(self.params.b1):
            self.params.b1[i] -= learning_rate * grads.b1[i]
        for i, row in enumerate(self.params.w2):
            for j, _ in enumerate(row):
                row[j] -= learning_rate * grads.w2[i][j]
        for i, value in enumerate(self.params.b2):
            self.params.b2[i] -= learning_rate * grads.b2[i]


class TransactionDiscriminator:
    """A discriminator that scores tabular samples."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        scale = 1.0 / math.sqrt(input_dim)
        self.params = LayerParameters(
            w1=random_matrix(hidden_dim, input_dim, scale),
            b1=random_vector(hidden_dim, scale),
            w2=random_matrix(1, hidden_dim, scale),
            b2=random_vector(1, scale),
        )

    def forward(self, vector: Vector) -> Tuple[float, DiscriminatorCache]:
        z1 = vector_add(matvec_mul(self.params.w1, vector), self.params.b1)
        h1 = [tanh(value) for value in z1]
        z2 = matvec_mul(self.params.w2, h1)[0] + self.params.b2[0]
        output = sigmoid(z2)
        cache = DiscriminatorCache(
            input_vector=vector,
            hidden_linear=z1,
            hidden_activation=h1,
            output_linear=z2,
            output_activation=output,
        )
        return output, cache

    def backward(self, cache: DiscriminatorCache, grad_logit: float) -> Tuple[Vector, LayerGradients]:
        grad_w2 = zeros_like(self.params.w2)
        grad_b2 = [grad_logit]
        grad_hidden = zeros_vector(len(cache.hidden_activation))

        for j, activation in enumerate(cache.hidden_activation):
            grad_w2[0][j] += grad_logit * activation
            grad_hidden[j] += grad_logit * self.params.w2[0][j]

        grad_hidden_linear = [grad_hidden[i] * tanh_derivative(cache.hidden_linear[i]) for i in range(len(grad_hidden))]

        grad_w1 = zeros_like(self.params.w1)
        grad_b1 = zeros_vector(len(self.params.b1))
        grad_input = zeros_vector(len(cache.input_vector))
        for i, grad in enumerate(grad_hidden_linear):
            grad_b1[i] = grad
            for j, input_value in enumerate(cache.input_vector):
                grad_w1[i][j] += grad * input_value
                grad_input[j] += grad * self.params.w1[i][j]

        grads = LayerGradients(w1=grad_w1, b1=grad_b1, w2=grad_w2, b2=grad_b2)
        return grad_input, grads

    def apply_gradients(self, grads: LayerGradients, learning_rate: float) -> None:
        for i, row in enumerate(self.params.w1):
            for j, _ in enumerate(row):
                row[j] -= learning_rate * grads.w1[i][j]
        for i, value in enumerate(self.params.b1):
            self.params.b1[i] -= learning_rate * grads.b1[i]
        for i, row in enumerate(self.params.w2):
            for j, _ in enumerate(row):
                row[j] -= learning_rate * grads.w2[i][j]
        self.params.b2[0] -= learning_rate * grads.b2[0]


class SyntheticDataGenerator:
    """Trainable GAN wrapper that produces synthetic tabular data."""

    def __init__(self, input_dim: int, config: GANTrainingConfig | None = None) -> None:
        self.config = config or GANTrainingConfig()
        self.generator = TransactionGenerator(
            self.config.noise_dim, input_dim, self.config.hidden_dim
        )
        self.discriminator = TransactionDiscriminator(
            input_dim, self.config.hidden_dim
        )

    def _noise_vector(self) -> Vector:
        return [random.gauss(0.0, 1.0) for _ in range(self.config.noise_dim)]

    def train(self, data: Sequence[Vector]) -> None:
        samples = [list(sample) for sample in data]
        if not samples:
            return

        for _ in range(self.config.epochs):
            random.shuffle(samples)
            for real_sample in samples:
                # Train discriminator
                noise = self._noise_vector()
                fake_sample, gen_cache = self.generator.forward(noise)
                real_output, real_cache = self.discriminator.forward(real_sample)
                fake_output, fake_cache = self.discriminator.forward(fake_sample)

                real_grad = real_output - 1.0
                fake_grad = fake_output

                _, real_grads = self.discriminator.backward(real_cache, real_grad)
                _, fake_grads = self.discriminator.backward(fake_cache, fake_grad)

                combined_grads = LayerGradients(
                    w1=[[r + f for r, f in zip(row_r, row_f)] for row_r, row_f in zip(real_grads.w1, fake_grads.w1)],
                    b1=[r + f for r, f in zip(real_grads.b1, fake_grads.b1)],
                    w2=[[r + f for r, f in zip(row_r, row_f)] for row_r, row_f in zip(real_grads.w2, fake_grads.w2)],
                    b2=[real_grads.b2[0] + fake_grads.b2[0]],
                )
                self.discriminator.apply_gradients(combined_grads, self.config.learning_rate)

                # Train generator
                noise = self._noise_vector()
                generated_sample, gen_cache = self.generator.forward(noise)
                fake_output, fake_cache = self.discriminator.forward(generated_sample)
                gen_grad_logit = fake_output - 1.0
                grad_input, disc_grads = self.discriminator.backward(fake_cache, gen_grad_logit)
                # Update discriminator with zero gradients to keep weights unchanged
                zero_grads = LayerGradients(
                    w1=zeros_like(disc_grads.w1),
                    b1=zeros_vector(len(disc_grads.b1)),
                    w2=zeros_like(disc_grads.w2),
                    b2=zeros_vector(len(disc_grads.b2)),
                )
                self.discriminator.apply_gradients(zero_grads, 0.0)

                gen_grads = self.generator.backward(gen_cache, grad_input)
                self.generator.apply_gradients(gen_grads, self.config.learning_rate)

    def generate(self, num_samples: int) -> List[Vector]:
        generated: List[Vector] = []
        for _ in range(num_samples):
            noise = self._noise_vector()
            sample, _ = self.generator.forward(noise)
            generated.append(sample)
        return generated


__all__ = [
    "GANTrainingConfig",
    "SyntheticDataGenerator",
    "TransactionDiscriminator",
    "TransactionGenerator",
]
