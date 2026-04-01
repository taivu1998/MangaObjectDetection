import unittest

from manga_detection.data.annotation_parsing import annotation_to_detection_target, annotation_to_yolo_rows, filter_annotation_contents


class AnnotationParsingTests(unittest.TestCase):
    def setUp(self):
        self.annotation = {
            "@width": 100,
            "@height": 200,
            "contents": [
                {"type": "body", "@xmin": 10, "@xmax": 20, "@ymin": 30, "@ymax": 60},
                {"type": "face", "@xmin": 5, "@xmax": 5, "@ymin": 10, "@ymax": 20},
            ],
        }

    def test_filter_annotation_contents_removes_invalid_boxes(self):
        cleaned = filter_annotation_contents(self.annotation)
        self.assertEqual(len(cleaned["contents"]), 1)
        self.assertEqual(cleaned["contents"][0]["type"], "body")

    def test_detection_target_shapes_remain_consistent(self):
        target = annotation_to_detection_target(self.annotation, image_id=7)
        self.assertEqual(tuple(target["boxes"].shape), (1, 4))
        self.assertEqual(tuple(target["labels"].shape), (1,))
        self.assertEqual(tuple(target["iscrowd"].shape), (1,))
        self.assertEqual(target["image_id"].item(), 7)

    def test_empty_detection_target_is_shaped_correctly(self):
        empty_annotation = {"@width": 100, "@height": 200, "contents": []}
        target = annotation_to_detection_target(empty_annotation)
        self.assertEqual(tuple(target["boxes"].shape), (0, 4))
        self.assertEqual(tuple(target["labels"].shape), (0,))
        self.assertEqual(tuple(target["iscrowd"].shape), (0,))

    def test_yolo_rows_skip_invalid_boxes(self):
        rows = annotation_to_yolo_rows(self.annotation)
        self.assertEqual(len(rows), 1)
        label, x_center, y_center, width, height = rows[0]
        self.assertEqual(label, 0)
        self.assertAlmostEqual(x_center, 0.15)
        self.assertAlmostEqual(y_center, 0.225)
        self.assertAlmostEqual(width, 0.10)
        self.assertAlmostEqual(height, 0.15)


if __name__ == "__main__":
    unittest.main()
