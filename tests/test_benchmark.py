"""ベンチマークの計測ロジックのテスト（APIキー不要）。"""

from types import SimpleNamespace

import pytest

from app.benchmark import run_benchmark


class FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, *, model, contents):
        self.calls.append((model, contents))
        return SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=4,
                total_token_count=14,
            )
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_run_benchmark_records_each_model_question_and_usage():
    client = FakeClient()
    results = run_benchmark(
        client,
        ["model-a", "model-b"],
        ["question"],
        repeats=2,
        clock=iter([1, 1.25, 2, 2.25, 3, 3.25, 4, 4.25]).__next__,
    )

    assert len(results) == 4
    assert [result.model for result in results] == [
        "model-a",
        "model-a",
        "model-b",
        "model-b",
    ]
    assert all(result.elapsed_seconds == pytest.approx(0.25) for result in results)
    assert all(result.total_tokens == 14 for result in results)
    assert client.models.calls == [
        ("model-a", "question"),
        ("model-a", "question"),
        ("model-b", "question"),
        ("model-b", "question"),
    ]


@pytest.mark.parametrize(
    ("repeats", "pause_seconds"),
    [(0, 0), (1, -1)],
)
def test_run_benchmark_rejects_invalid_options(repeats, pause_seconds):
    with pytest.raises(ValueError):
        run_benchmark(FakeClient(), ["model"], ["question"], repeats=repeats, pause_seconds=pause_seconds)
