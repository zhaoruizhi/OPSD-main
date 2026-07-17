import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class QuickJsonlMergeTests(unittest.TestCase):
    def test_merge_jsonl_files_streams_valid_records(self):
        from eval.quick_jsonl_merge import iter_jsonl_records, merge_jsonl_files

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            second.write_text('{"id": 3}\n', encoding="utf-8")

            count = merge_jsonl_files([first, second], output)

            self.assertEqual(count, 3)
            self.assertEqual([row["id"] for row in iter_jsonl_records(output)], [1, 2, 3])

    def test_merge_jsonl_files_can_sort_small_ranked_outputs(self):
        from eval.quick_jsonl_merge import iter_jsonl_records, merge_jsonl_files

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            output = root / "merged.jsonl"
            first.write_text('{"rank": 3}\n{"rank": 1}\n', encoding="utf-8")
            second.write_text('{"rank": 2}\n', encoding="utf-8")

            merge_jsonl_files([first, second], output, sort_key="rank")

            self.assertEqual([row["rank"] for row in iter_jsonl_records(output)], [1, 2, 3])

    def test_corrupt_input_does_not_replace_existing_output(self):
        from eval.quick_jsonl_merge import merge_jsonl_files

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            bad = root / "bad.jsonl"
            output = root / "merged.jsonl"
            bad.write_text('{"id": 1}\n{"id":\n', encoding="utf-8")
            output.write_text('{"kept": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"bad\.jsonl:2"):
                merge_jsonl_files([bad], output)

            self.assertEqual(output.read_text(encoding="utf-8"), '{"kept": true}\n')

    def test_non_object_record_reports_file_and_line(self):
        from eval.quick_jsonl_merge import iter_jsonl_records

        with TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "list.jsonl"
            source.write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"list\.jsonl:1"):
                list(iter_jsonl_records(source))


if __name__ == "__main__":
    unittest.main()
