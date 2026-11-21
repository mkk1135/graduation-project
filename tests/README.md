# Tests

Unit tests of the project's own logic: metrics, Wilson interval, McNemar test, label mapping and large-image rule (evaluate.py), the decision rules (decide.py, also against the stored results), the rescaled score of the combined detector (fusion.py), the checker (second_opinion.py), what the page of app.py shows with and without a checker and its per-visitor rate limit, and the helpers of results_db.py, including the detector left out of the written comparison.
They run on the CPU in a few seconds and download or load no model: detectors, Gradio and the Claude client are replaced by stand-ins.
Run them from the project folder with `pip install pillow pytest anthropic` and `python -m pytest tests`; the GitHub workflow (`.github/workflows/checks.yml`) runs them on every push.
