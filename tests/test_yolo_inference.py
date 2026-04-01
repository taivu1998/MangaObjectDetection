import tempfile
import unittest
from pathlib import Path

from PIL import Image

from manga_detection.integrations.yolo import build_detect_parser, load_class_names, render_detections_to_image


class YoloInferenceTests(unittest.TestCase):
    def test_detect_parser_defaults_to_output_dir(self):
        args = build_detect_parser().parse_args([])
        self.assertEqual(args.backend, "native")
        self.assertEqual(args.output_dir, Path("output"))
        self.assertEqual(args.conf_thres, 0.5)
        self.assertEqual(args.nms_thres, 0.4)

    def test_load_class_names_strips_empty_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            classes_path = Path(tmpdir) / "classes.names"
            classes_path.write_text("body\n\nface\n")
            self.assertEqual(load_class_names(classes_path), ["body", "face"])

    def test_render_detections_to_image_saves_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            image_path = tmpdir_path / "sample.jpg"
            output_path = tmpdir_path / "output" / "sample.png"
            Image.new("RGB", (64, 64), color=(255, 255, 255)).save(image_path)

            render_detections_to_image(
                image_path=image_path,
                detections=[(5.0, 5.0, 30.0, 30.0, 0.95, 0)],
                class_names=["body", "face", "frame", "text"],
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            with Image.open(output_path) as rendered:
                self.assertEqual(rendered.size, (64, 64))


if __name__ == "__main__":
    unittest.main()
