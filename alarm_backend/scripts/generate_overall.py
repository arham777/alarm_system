import os
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pvcI_health_monitor import compute_pvcI_overall_health, DEFAULT_CONFIG
from config import PVCI_FOLDER


def main() -> None:
    result = compute_pvcI_overall_health(
        PVCI_FOLDER,
        DEFAULT_CONFIG,
        max_workers=12,
        per_file_timeout=None,
        include_details=True,
        limit_unhealthy_per_source=50,
    )

    out_dir = os.path.join('PVCI-overall-health')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pvcI-overall-health.json')

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('Wrote', out_path, 'size', os.path.getsize(out_path))


if __name__ == '__main__':
    main()


