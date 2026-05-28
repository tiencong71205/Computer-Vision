"""
src/utils/config.py
Đọc config.yaml và trả về dict có thể access bằng dấu chấm.

Dùng:
    from src.utils.config import load_config
    cfg = load_config()                      # đọc config.yaml mặc định
    cfg = load_config('config.custom.yaml')  # đọc file khác

    # Access bình thường
    cfg['fire']['conf']          # → 0.45

    # Hoặc dùng dấu chấm
    cfg.fire.conf                # → 0.45
    cfg.models.lstm.hidden       # → 64
"""

import yaml
from pathlib import Path


class DotDict(dict):
    """Dict có thể access bằng dấu chấm: d.key thay vì d['key']."""
    def __getattr__(self, key):
        try:
            val = self[key]
            return DotDict(val) if isinstance(val, dict) else val
        except KeyError:
            raise AttributeError(f"Config không có key: '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str = "config.yaml") -> DotDict:
    """
    Đọc file YAML và trả về DotDict.

    Args:
        path: đường dẫn tới file config (mặc định: config.yaml)

    Returns:
        DotDict — config có thể access bằng cfg.key.subkey
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {cfg_path.resolve()}")

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return DotDict(raw)
