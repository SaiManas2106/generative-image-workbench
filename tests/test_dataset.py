import unittest

from genimage_workbench.dataset import DatasetRecord, validate_manifest


class DatasetValidationTests(unittest.TestCase):
    def test_reports_duplicates_and_missing_validation_split(self):
        report = validate_manifest([
            DatasetRecord("images/a.png", "hero concept", "train", "hero"),
            DatasetRecord("images/a.png", "hero pose", "train", "hero"),
        ])

        self.assertEqual(report.duplicate_paths, ["images/a.png"])
        self.assertIn("manifest requires at least one validation record", report.errors)

    def test_reports_identity_coverage(self):
        report = validate_manifest([
            DatasetRecord("images/a.png", "hero", "train", "hero"),
            DatasetRecord("images/b.png", "scene", "validation"),
        ])

        self.assertEqual(report.identity_coverage, 0.5)
        self.assertEqual(report.errors, [])
