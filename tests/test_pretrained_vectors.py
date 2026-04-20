import unittest
from unittest.mock import patch

from pretrained_vectors import load_pretrained_vectors_with_recovery


class _FakeApi:
    def __init__(self, base_dir, failures):
        self.BASE_DIR = str(base_dir)
        self._failures = list(failures)
        self.calls = 0

    def load(self, model_name):
        self.calls += 1
        if self._failures:
            exc = self._failures.pop(0)
            if exc is not None:
                raise exc
        return {"model": model_name, "loaded": True}


class PretrainedVectorsTests(unittest.TestCase):
    def test_retries_and_recovers_from_load_data_attribute_error(self):
        model_name = "glove-wiki-gigaword-50"
        api = _FakeApi(
            base_dir="unused",
            failures=[AttributeError("module 'glove-wiki-gigaword-50' has no attribute 'load_data'"), None],
        )

        fallback_called = {"value": False}

        def fallback():
            fallback_called["value"] = True
            return {"fallback": True}

        with patch("pretrained_vectors._clear_cached_model_files", return_value=True) as clear_mock:
            result = load_pretrained_vectors_with_recovery(
                api_module=api,
                model_name=model_name,
                fallback_builder=fallback,
                print_fn=lambda *_: None,
            )

        self.assertEqual(api.calls, 2)
        self.assertEqual(result["model"], model_name)
        self.assertFalse(fallback_called["value"])
        clear_mock.assert_called_once_with(api, model_name)

    def test_falls_back_when_retry_still_fails(self):
        model_name = "glove-wiki-gigaword-50"
        api = _FakeApi(
            base_dir="unused",
            failures=[
                AttributeError("module 'glove-wiki-gigaword-50' has no attribute 'load_data'"),
                RuntimeError("network down"),
            ],
        )

        with patch("pretrained_vectors._clear_cached_model_files", return_value=True) as clear_mock:
            result = load_pretrained_vectors_with_recovery(
                api_module=api,
                model_name=model_name,
                fallback_builder=lambda: {"fallback": True},
                print_fn=lambda *_: None,
            )

        self.assertEqual(api.calls, 2)
        self.assertEqual(result, {"fallback": True})
        clear_mock.assert_called_once_with(api, model_name)

    def test_falls_back_on_non_attribute_error_without_retry(self):
        model_name = "glove-wiki-gigaword-50"
        api = _FakeApi(
            base_dir="unused",
            failures=[RuntimeError("generic failure")],
        )

        with patch("pretrained_vectors._clear_cached_model_files", return_value=True) as clear_mock:
            result = load_pretrained_vectors_with_recovery(
                api_module=api,
                model_name=model_name,
                fallback_builder=lambda: {"fallback": True},
                print_fn=lambda *_: None,
            )

        self.assertEqual(api.calls, 1)
        self.assertEqual(result, {"fallback": True})
        clear_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
