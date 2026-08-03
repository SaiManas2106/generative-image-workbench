import unittest

from genimage_workbench.plans import create_adapter_plan, create_inference_plan
from genimage_workbench.runtime import detect_gpu_profile


class PlanningTests(unittest.TestCase):
    def test_adapter_plan_has_reproducible_defaults(self):
        plan = create_adapter_plan("characters-v1", "portrait of <hero>")

        self.assertEqual(plan.rank, 16)
        self.assertEqual(plan.seed, 42)
        self.assertEqual(plan.precision, "bf16")

    def test_inference_downgrades_resolution_for_vram_budget(self):
        plan = create_inference_plan("portrait of explorer", "explorer", available_vram_gb=8)

        self.assertEqual(plan.width, 768)
        self.assertEqual(plan.height, 768)
        self.assertEqual(plan.required_vram_gb, 7.5)

    def test_inference_rejects_too_little_vram(self):
        with self.assertRaisesRegex(ValueError, "at least 6 GB"):
            create_inference_plan("portrait", None, available_vram_gb=4)

    def test_runtime_probe_selects_bf16_for_large_gpu(self):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_device_properties(index):
                return type("Properties", (), {"total_memory": 24 * 1024 ** 3})()

        profile = detect_gpu_profile(type("FakeTorch", (), {"cuda": FakeCuda})())
        self.assertEqual(profile.device, "cuda:0")
        self.assertEqual(profile.preferred_precision, "bf16")
