import json

import pytest

from fastfill.v2.batch import TinyTokenizer
from fastfill.v2.model import ModelConfig, build_model
from fastfill.v2.predict import main
from fastfill.v2.schema import validate_layout
from fastfill.v2.tests.test_execution import sample


def test_condition_only_prediction_cli_no_targets(tmp_path):
    model_dir = tmp_path / "model"
    model = build_model(ModelConfig(backbone="tiny", lora_rank=0, decoder_dim=16,
                        decoder_heads=2, decoder_layers=1, tiny_hidden_size=16))
    model.save_pretrained(model_dir)
    TinyTokenizer().save_pretrained(tmp_path / "tokenizer")
    condition = sample()["condition"]
    request = tmp_path / "request.json"
    request.write_text(json.dumps(condition))
    output = tmp_path / "prediction.json"
    main(["--checkpoint", str(model_dir), "--condition", str(request), "--output", str(output)])
    validate_layout(json.loads(output.read_text()), condition)
    with pytest.raises(FileExistsError):
        main(["--checkpoint", str(model_dir), "--condition", str(request), "--output", str(request)])
    with pytest.raises(ValueError):
        main(["--checkpoint", str(model_dir), "--condition", str(request), "--output",
              "/Volumes/harddisk/3D_Room_Collections/generated.json"])
