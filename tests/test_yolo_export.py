import tempfile
import unittest
from pathlib import Path

import pandas as pd

from manga_detection.data.yolo_export import (
    build_manifest_lines,
    export_labels,
    image_path_to_export_name,
    image_path_to_label_name,
    write_manifest,
)


class YoloExportTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "image_path": "./Manga109/images/BookA/001.jpg",
                    "image_annotation": {
                        "@width": 100,
                        "@height": 200,
                        "contents": [{"type": "body", "@xmin": 10, "@xmax": 20, "@ymin": 30, "@ymax": 60}],
                    },
                }
            ]
        )

    def test_image_name_mapping(self):
        self.assertEqual(image_path_to_export_name("./Manga109/images/BookA/001.jpg"), "BookA-001.jpg")
        self.assertEqual(image_path_to_label_name("./Manga109/images/BookA/001.jpg"), "BookA-001.txt")

    def test_export_labels_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            metadata_path = tmpdir_path / "split.pkl"
            self.df.to_pickle(metadata_path)

            labels_dir = tmpdir_path / "labels"
            export_labels(metadata_path, labels_dir)
            self.assertTrue((labels_dir / "BookA-001.txt").exists())

            manifest_path = tmpdir_path / "train.txt"
            write_manifest(metadata_path, manifest_path)
            self.assertEqual(manifest_path.read_text().strip(), "data/custom/images/BookA-001.jpg")
            self.assertEqual(build_manifest_lines(metadata_path), ["data/custom/images/BookA-001.jpg"])


if __name__ == "__main__":
    unittest.main()
