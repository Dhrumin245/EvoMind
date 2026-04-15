import unittest
from pathlib import Path

from api.trainer import EvoTrainer


class ResumeCheckpointPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trainer = object.__new__(EvoTrainer)
        fixture_root = Path("tests/fixtures/resume_checkpoints")
        cls.trainer.checkpoint_dir = fixture_root.resolve(strict=True)
        cls.valid_checkpoint = next(
            cls.trainer.checkpoint_dir.glob("checkpoint_gen_*.json")
        )
        cls.outside_checkpoint_file = (fixture_root.parent / "outside-config.json").resolve(strict=True)

    def test_relative_checkpoint_path_inside_directory_is_allowed(self) -> None:
        resolved = self.trainer.resolve_checkpoint_path(self.valid_checkpoint.name)

        self.assertEqual(resolved, self.valid_checkpoint)

    def test_absolute_checkpoint_path_inside_directory_is_allowed(self) -> None:
        resolved = self.trainer.resolve_checkpoint_path(str(self.valid_checkpoint))

        self.assertEqual(resolved, self.valid_checkpoint)

    def test_relative_traversal_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Checkpoint path must stay within the job checkpoint directory",
        ):
            self.trainer.resolve_checkpoint_path("../config.json")

    def test_absolute_path_outside_directory_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Checkpoint path must stay within the job checkpoint directory",
        ):
            self.trainer.resolve_checkpoint_path(str(self.outside_checkpoint_file))


if __name__ == "__main__":
    unittest.main()
