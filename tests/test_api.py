import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loads():
    """Test that model file exists and loads correctly."""
    import torch
    from src.model import build_model
    
    assert os.path.exists('models/best_model.pth'), \
        "Model file not found"
    
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(
        torch.load('models/best_model.pth', 
                  map_location='cpu')
    )
    model.eval()
    print("Model loads successfully")

def test_model_inference():
    """Test model produces correct output shape."""
    import torch
    from src.model import build_model
    
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(
        torch.load('models/best_model.pth',
                  map_location='cpu')
    )
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2), \
        f"Expected output shape (1, 2), got {output.shape}"
    print(f"Inference output shape: {output.shape}")

def test_data_pipeline():
    """Test data loading works correctly."""
    from src.data_prep import get_dataloaders
    
    train_loader, test_loader, classes = get_dataloaders()
    
    assert len(classes) == 2, "Expected 2 classes"
    assert 'def_front' in classes, "Missing def_front class"
    assert 'ok_front' in classes, "Missing ok_front class"
    assert len(test_loader.dataset) > 0, "Test dataset is empty"
    
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_loader.dataset)}")

def test_api_response_structure():
    """Test API returns correct response structure."""
    from fastapi.testclient import TestClient
    import io
    from PIL import Image
    
    # Import app
    sys.path.insert(0, 'api')
    from main import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'operational'
    assert data['model_loaded'] == True
    print("Health check passed")
    
    # Test model info endpoint
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert 'performance' in data
    print("Model info endpoint passed")