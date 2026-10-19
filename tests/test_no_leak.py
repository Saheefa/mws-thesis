"""The pipeline loader must never load Train ground truth."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data


def test_pipeline_loader_hides_y_hidden(tmp_path):
    df = data.clean(data.pd.DataFrame(
        [(f"cb headline {i}", 1) for i in range(200)] + [(f"news headline {i}", 0) for i in range(200)],
        columns=["title", "label"]))
    data.save_splits(data.make_splits(df, dev_per_class=20), tmp_path)
    sp = data.load_pipeline_splits(tmp_path)
    assert list(sp["train"].columns) == ["title"]
    assert "label" in sp["dev"] and "label" in sp["test"]
