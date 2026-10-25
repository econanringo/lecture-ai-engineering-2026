"""複数のモデルと質問を比較するベンチマーク。

使い方:
    uv run python -m app.benchmark \
        --models gemini-3.5-flash-lite,gemini-3.5-flash \
        --questions "LoRAとは何ですか？" "RAGとは何ですか？"

各試行の結果を標準出力に表示し、必要なら JSON または CSV に保存する。
API のレート制限に配慮するため、試行間隔は --pause で指定できる。
"""

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from google import genai
from google.genai import errors

from app.ask import DEFAULT_QUESTION
from app.config import api_error_message, get_api_key


@dataclass
class BenchmarkResult:
    """1回の API 呼び出しの計測結果。"""

    model: str
    question: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    error: str | None = None


def _usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if value is not None else None


def run_benchmark(
    client: Any,
    models: Iterable[str],
    questions: Iterable[str],
    *,
    repeats: int = 1,
    pause_seconds: float = 0,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> list[BenchmarkResult]:
    """指定した組み合わせを計測する。API エラーも結果として記録して続行する。"""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if pause_seconds < 0:
        raise ValueError("pause_seconds must not be negative")

    results: list[BenchmarkResult] = []
    combinations = [
        (model, question, repeat)
        for model in models
        for question in questions
        for repeat in range(repeats)
    ]
    for index, (model, question, _) in enumerate(combinations):
        started = clock()
        try:
            response = client.models.generate_content(model=model, contents=question)
            usage = response.usage_metadata
            input_tokens = _usage_value(usage, "prompt_token_count")
            output_tokens = _usage_value(usage, "candidates_token_count")
            total_tokens = _usage_value(usage, "total_token_count")
            error = None
        except errors.APIError as exc:
            input_tokens = output_tokens = total_tokens = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = clock() - started
        results.append(
            BenchmarkResult(
                model=model,
                question=question,
                elapsed_seconds=round(elapsed, 6),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                error=error,
            )
        )
        if index + 1 < len(combinations) and pause_seconds:
            sleeper(pause_seconds)
    return results


def _write_results(results: list[BenchmarkResult], output: Path, output_format: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    if output_format == "json":
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _parse_models(value: str) -> list[str]:
    models = [model.strip() for model in value.split(",") if model.strip()]
    if not models:
        raise argparse.ArgumentTypeError("モデルを1つ以上指定してください")
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="複数のモデルと質問の応答時間・トークン数を比較する")
    parser.add_argument("--models", type=_parse_models, required=True, help="カンマ区切りのモデル名")
    parser.add_argument(
        "--questions",
        nargs="+",
        default=[DEFAULT_QUESTION],
        help="比較する質問（複数指定可）",
    )
    parser.add_argument("--repeat", type=int, default=1, help="各組み合わせの実行回数（既定: 1）")
    parser.add_argument("--pause", type=float, default=0, help="試行間の待ち時間（秒）")
    parser.add_argument("--output", type=Path, help="結果の保存先")
    parser.add_argument("--format", choices=("json", "csv"), default="json", dest="output_format")
    args = parser.parse_args()

    client = genai.Client(api_key=get_api_key())
    results = run_benchmark(
        client,
        args.models,
        args.questions,
        repeats=args.repeat,
        pause_seconds=args.pause,
    )
    print("model,question,elapsed_seconds,input_tokens,output_tokens,total_tokens,error")
    for result in results:
        print(
            f"{result.model},{json.dumps(result.question, ensure_ascii=False)},"
            f"{result.elapsed_seconds},{result.input_tokens},{result.output_tokens},"
            f"{result.total_tokens},{json.dumps(result.error, ensure_ascii=False)}"
        )
    if args.output:
        _write_results(results, args.output, args.output_format)
        print(f"[saved] {args.output}")
    if any(result.error for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except errors.APIError as exc:
        print(api_error_message(exc.code, exc.message))
        raise SystemExit(1)
