from crib_ai_trainer2.model_registry import save_model_artifact
import os
import json

def test_save_model_artifact(tmp_path):
    model_name = "testmodel"
    version = 1
    model_data = {"w": [1,2,3], "b": [0.1]}
    metrics = {"accuracy": 0.99}
    config = {"lr": 0.01}
    base_dir = tmp_path
    save_model_artifact(model_name, version, model_data, metrics, config, base_dir=str(base_dir))
    out_dir = base_dir / model_name / f"v{version}"
    best_path = out_dir / "best.json"
    assert best_path.exists()
    with open(best_path) as f:
        data = json.load(f)
    assert data["model"] == model_data
    assert data["metrics"] == metrics
    assert data["config"] == config
    hist_dir = out_dir / "history"
    assert any(hist_dir.iterdir())
