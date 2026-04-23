import re
from pathlib import Path


def _decode_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            pass
    return data.decode("utf-8", errors="replace")


_RE_PROXY = re.compile(r"^\s*proxy=([0-9]+(?:\.[0-9]+)?)\b", re.MULTILINE)
_RE_BENCH = re.compile(r"\bibm\d\d\b")


def _bench_from_path(path: Path) -> str | None:
    m = _RE_BENCH.search(path.name.lower())
    return m.group(0) if m else None


def parse_eval_file(path: Path) -> tuple[str | None, float | None]:
    text = _decode_text(path)
    bench = _bench_from_path(path) or (_RE_BENCH.search(text.lower()).group(0) if _RE_BENCH.search(text.lower()) else None)
    m = _RE_PROXY.search(text)
    proxy = float(m.group(1)) if m else None
    return bench, proxy


def main() -> None:
    from macro_place.evaluate import REPLACE_BASELINES

    runs_dir = Path("runs")
    files = sorted(runs_dir.glob("eval-*.txt"))
    per_bench: dict[str, list[tuple[float, str]]] = {}
    for f in files:
        bench, proxy = parse_eval_file(f)
        if bench is None or proxy is None:
            continue
        per_bench.setdefault(bench, []).append((proxy, f.name))

    benches = sorted(per_bench.keys())
    if not benches:
        print("No eval logs found under runs/eval-*.txt")
        return

    rows = []
    for b in benches:
        best_proxy, best_file = min(per_bench[b], key=lambda x: x[0])
        base = float(REPLACE_BASELINES.get(b, float("nan")))
        delta = best_proxy - base
        rows.append((b, best_proxy, base, delta, best_file))

    print(f"benches={len(rows)} logs={sum(len(v) for v in per_bench.values())}")
    print("bench  best_proxy  replace  delta    best_log")
    for b, p, base, d, fn in rows:
        print(f"{b:5s}  {p:9.4f}  {base:7.4f}  {d:+7.4f}  {fn}")

    avg_proxy = sum(r[1] for r in rows) / len(rows)
    avg_base = sum(r[2] for r in rows) / len(rows)
    print("")
    print(f"avg_proxy={avg_proxy:.4f} avg_replace={avg_base:.4f} delta={avg_proxy-avg_base:+.4f}")


if __name__ == "__main__":
    main()
