import importlib.util
import unittest
from pathlib import Path

from eval.quick_first_error_ablation import (
    build_first_error_condition_specs,
    segment_kl_summary,
    summarize_kl_slice,
)
from eval.quick_opsd_common import (
    build_first_error_user_message,
    build_reference_user_message,
    first_error_text_slices,
    first_error_token_ranges,
    reasoning_step_spans,
    validate_first_error_diagnostic,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)


def valid_diagnostic():
    return {
        "prefix_valid_until": "Step one valid.",
        "first_error_sentence": "Step two wrong.",
        "error_type": "algebraic_error",
        "valid_prefix_summary": "The first step is valid.",
        "student_plan": "Use algebra.",
        "local_repair": "Fix the sign.",
        "next_subgoal_after_repair": "Continue with the corrected sign.",
    }


class FirstErrorAblationTests(unittest.TestCase):
    def test_generator_diagnostic_schema_requires_sentence_fields(self):
        module_path = Path(__file__).resolve().parents[1] / "generate_1st-error_json.py"
        spec = importlib.util.spec_from_file_location("generate_1st_error_json", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        self.assertEqual(module.validate_diagnostic(valid_diagnostic()), [])

        missing = dict(valid_diagnostic())
        del missing["first_error_sentence"]
        self.assertTrue(any("missing fields" in error for error in module.validate_diagnostic(missing)))

        invalid_none = dict(valid_diagnostic())
        invalid_none["error_type"] = "none"
        invalid_none["local_repair"] = ""
        invalid_none["next_subgoal_after_repair"] = None
        self.assertIn(
            'first_error_sentence must be null when error_type is "none"',
            module.validate_diagnostic(invalid_none),
        )

    def test_common_first_error_validation_rejects_old_schema(self):
        old = dict(valid_diagnostic())
        old["first_error_span"] = old.pop("first_error_sentence")

        errors = validate_first_error_diagnostic(old)

        self.assertTrue(any("regenerate first-error diagnostics with the sentence schema" in error for error in errors))

    def test_first_error_prefix_and_neighborhood_ranges(self):
        full_generation = "Step one valid.\n\nStep two wrong.\n\nStep three later."
        diagnostic = valid_diagnostic()

        spans = reasoning_step_spans(full_generation)
        text_slices = first_error_text_slices(full_generation, diagnostic)
        token_ranges = first_error_token_ranges(
            FakeTokenizer(),
            full_generation,
            diagnostic,
            neighborhood_before=1,
            neighborhood_after=1,
        )

        self.assertEqual([span["text"] for span in spans], ["Step one valid.", "Step two wrong.", "Step three later."])
        self.assertEqual(text_slices["student_prefix"], "Step one valid.")
        self.assertTrue(text_slices["target_tail_text"].startswith("\n\nStep two wrong."))
        self.assertEqual(full_generation[text_slices["first_error_char_range"][0] : text_slices["first_error_char_range"][1]], "Step two wrong.")
        self.assertEqual(token_ranges["valid_prefix_range"], [0, 3])
        self.assertEqual(token_ranges["first_error_range"], [3, 6])
        self.assertEqual(token_ranges["first_error_neighborhood_range"], [2, 7])

    def test_first_error_prompts_use_reference_template(self):
        diagnostic_prompt = build_first_error_user_message("Compute 2+2.", valid_diagnostic(), ground_truth="4")
        text_prompt = build_reference_user_message("Compute 2+2.", "A reference solution.", ground_truth="4")

        self.assertIn("Final answer: 4", diagnostic_prompt)
        self.assertLess(diagnostic_prompt.index("Final answer: 4"), diagnostic_prompt.index("Reference Solution Begin"))
        self.assertIn('"first_error_sentence": "Step two wrong."', diagnostic_prompt)
        self.assertIn("A reference solution.", text_prompt)

    def test_first_error_condition_specs_match_plan(self):
        specs = {spec.name: spec for spec in build_first_error_condition_specs()}

        self.assertEqual(list(specs), ["teacher_base_w_text", "teacher_base_w_first_error"])
        self.assertTrue(specs["teacher_base_w_text"].enable_thinking)
        self.assertEqual(specs["teacher_base_w_text"].prompt_kind, "reference_text")
        self.assertEqual(specs["teacher_base_w_first_error"].prompt_kind, "first_error_json")

    def test_segment_kl_summary_slices_token_metrics(self):
        record = {
            "kl_per_token": [1.0, 2.0, 3.0, 4.0],
            "delta_logp_target_per_token": [0.1, 0.2, 0.3, 0.4],
            "teacher_entropy_per_token": [1.0, 1.0, 2.0, 2.0],
            "student_entropy_per_token": [0.5, 0.5, 1.0, 1.0],
            "delta_entropy_per_token": [0.5, 0.5, 1.0, 1.0],
        }
        case = {
            "valid_prefix_range": [0, 2],
            "first_error_neighborhood_range": [1, 4],
        }

        valid_prefix = summarize_kl_slice(record, [0, 2])
        segments = segment_kl_summary(record, case)

        self.assertEqual(valid_prefix["num_tokens"], 2)
        self.assertEqual(valid_prefix["sum_kl"], 3.0)
        self.assertEqual(valid_prefix["mean_kl"], 1.5)
        self.assertEqual(segments["first_error_neighborhood"]["sum_kl"], 9.0)
        self.assertEqual(segments["first_error_neighborhood"]["mean_kl"], 3.0)


if __name__ == "__main__":
    unittest.main()
