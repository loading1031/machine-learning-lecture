import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 다운샘플링 함수
def downsample_images(images, scale=4):
    """
    이미지를 다운샘플링합니다.
    """
    low_res_images = []
    for img in images:
        low_res = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
        # low_res = cv2.resize(low_res, (img.shape[1], img.shape[0]))
        low_res_images.append(low_res)
    return np.array(low_res_images)

# ESPCN 모델 정의
def build_espcn(scale):
    """
    Efficient Sub-Pixel Convolutional Network 생성
    """
    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(None, None, 3)))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(3 * (scale ** 2), (3, 3), padding='same'))
    model.add(layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale)))
    return model

# 데이터 로드 (ImageNet)
def load_imagenet_sample():
    """
    ImageNet 샘플 데이터를 로드합니다.
    """
    # ImageNet 데이터는 일반적으로 직접 다운로드해야 함
    # 여기서는 대체 데이터로 CIFAR-10을 사용 (데모 목적으로)
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0  # 정규화
    x_test = x_test / 255.0
    return x_train, x_test

# Fashion-MNIST 데이터 로드
def load_fashion_mnist():
    """
    Fashion-MNIST 데이터를 로드합니다.
    """
    (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0  # 정규화 및 채널 추가
    x_test = np.expand_dims(x_test, axis=-1) / 255.0
    return x_train, x_test

# 학습
def train_espcn():
    """
    ESPCN 모델 학습
    """
    # 데이터 로드
    x_train, _ = load_imagenet_sample()
    x_train_low_res = downsample_images(x_train)

    # 모델 생성
    scale = 4
    model = build_espcn(scale)
    model.compile(optimizer='adam', loss='mse')

    # 학습
    model.fit(x_train_low_res, x_train, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

    return model

# 테스트
def test_on_fashion_mnist(model):
    """
    Fashion-MNIST에서 초해상도 테스트
    """
    _, x_test = load_fashion_mnist()

    # 다운샘플링 및 변환
    x_test_low_res = downsample_images(x_test, scale=4)
    x_test_sr = model.predict(x_test_low_res)

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test_low_res[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(3, 5, i + 6)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(3, 5, i + 11)
        plt.imshow(x_test_sr[i].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# 컬러라이제이션 (추가 작업)
def colorize_fashion_mnist(model):
    """
    Fashion-MNIST 데이터에 컬러라이제이션 적용
    """
    _, x_test = load_fashion_mnist()

    # 흑백 이미지를 컬러로 변환
    x_test_sr = model.predict(np.expand_dims(x_test, axis=-1))
    colorized_images = x_test_sr

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.axis('off')
        plt.subplot(2, 5, i + 6)
        plt.imshow(colorized_images[i].squeeze())
        plt.axis('off')
    plt.show()

def train_espcn_with_limited_data():
    """
    ESPCN 모델 학습 (10개의 제한된 데이터로)
    """
    # 데이터 로드
    x_train, _ = load_imagenet_sample()

    # 학습 데이터 제한 (10개만 사용)
    x_train = x_train[:10]
    x_train_low_res = downsample_images(x_train)

    # 모델 생성
    scale = 4
    model = build_espcn(scale)
    model.compile(optimizer='adam', loss='mse')

    # 학습
    model.fit(x_train_low_res, x_train, epochs=5, batch_size=2, validation_split=0.2)

    return model

def test_with_limited_data(model):
    """
    Fashion-MNIST에서 제한된 테스트 데이터로 초해상도 테스트
    """
    # 테스트 데이터 로드
    _, x_test = load_fashion_mnist()

    # 테스트 데이터 제한 (5개만 사용)
    x_test = x_test[:5]
    x_test_low_res = downsample_images(x_test, scale=4)
    x_test_sr = model.predict(x_test_low_res)

    # 결과 시각화
    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test_low_res[i].squeeze(), cmap='gray')
        plt.title("Low-Res")
        plt.axis('off')

        plt.subplot(3, 5, i + 6)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')

        plt.subplot(3, 5, i + 11)
        plt.imshow(x_test_sr[i].squeeze(), cmap='gray')
        plt.title("Super-Res")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 실행
model = train_espcn_with_limited_data()
test_with_limited_data(model)
