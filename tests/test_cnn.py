import torch

from src.models.cnn_feature_extractor import CNNFeatureExtractor


def test_cnn_feature_extraction():

    model = CNNFeatureExtractor()
    model.eval()

    # Simulate 20 preprocessed video frames
    x = torch.rand(20, 3, 224, 224)

    with torch.no_grad():
        features = model(x)

    print("\nInput shape:", x.shape)
    print("Feature shape:", features.shape)

    assert features.shape == (20, 512)