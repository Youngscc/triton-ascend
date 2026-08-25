import csv
from pathlib import Path
import tempfile
import unittest

from experiment_operators import compare_ub_csv

FIELDS = [
    "序号",
    "enable_dynamic_cv_pipeline",
    "intra_cache_num",
    "enable_auto_multi_buffer",
    "multibuffer_num",
    "vf_merge_level",
    "UB使用_KiB",
]


class CompareUbCsvTest(unittest.TestCase):

    @staticmethod
    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def row(number: int, intra, multibuffer, merge: int, ub) -> dict:
        return {
            "序号": number,
            "enable_dynamic_cv_pipeline": intra != "off",
            "intra_cache_num": intra,
            "enable_auto_multi_buffer": multibuffer != "off",
            "multibuffer_num": multibuffer,
            "vf_merge_level": merge,
            "UB使用_KiB": ub,
        }

    def test_compares_only_matching_parameter_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write_csv(first, [
                self.row(1, "off", "off", 0, "64"),
                self.row(2, 1, 2, 0, "65"),
                self.row(3, 2, 2, 0, ""),
                self.row(4, 3, 2, 0, "70"),
            ])
            self.write_csv(second, [
                self.row(1, "off", "off", 0, "64.0"),
                self.row(2, 1, 2, 0, "66"),
                self.row(3, 2, 2, 0, "68"),
                self.row(4, 4, 2, 0, "71"),
            ])

            result = compare_ub_csv.compare(first, second)

        self.assertEqual(result["matching"], 3)
        self.assertEqual(result["same"], 1)
        self.assertEqual(len(result["different"]), 1)
        self.assertEqual(result["missing_ub"], 1)
        self.assertEqual(result["only_first"], 1)
        self.assertEqual(result["only_second"], 1)

    def test_rejects_duplicate_parameter_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            duplicate = self.row(1, 1, 2, 0, "64")
            self.write_csv(first, [duplicate, {**duplicate, "序号": 2}])
            self.write_csv(second, [duplicate])

            with self.assertRaisesRegex(compare_ub_csv.ComparisonError, "duplicate parameter set"):
                compare_ub_csv.compare(first, second)

    def test_matches_old_and_new_dynamic_cv_axis_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            self.write_csv(first, [self.row(1, 3, 2, 1, "64")])
            new_fields = ["buf_slot_num_of_veccore" if field == "intra_cache_num" else field for field in FIELDS]
            with second.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=new_fields)
                writer.writeheader()
                row = self.row(1, 3, 2, 1, "64.0")
                row["buf_slot_num_of_veccore"] = row.pop("intra_cache_num")
                writer.writerow(row)

            result = compare_ub_csv.compare(first, second)

        self.assertEqual(result["key_fields"][:2], ["enable_dynamic_cv_pipeline", "dynamic_cv_slots"])
        self.assertEqual(result["matching"], 1)
        self.assertEqual(result["same"], 1)


if __name__ == "__main__":
    unittest.main()
