"""Command-line interface: ``aidisclose <command> --help``."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from ._version import __version__

log = logging.getLogger("aidisclose")


def _warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr)


def _same_file_path(a: str | Path, b: str | Path) -> bool:
    left, right = Path(a), Path(b)
    if os.path.normcase(str(left.resolve())) == os.path.normcase(str(right.resolve())):
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def _check_output_paths(outputs, *, inputs=(), result_dirs=()) -> None:
    """Reject aliases between outputs, inputs and protected result artifacts."""
    outputs = [(name, Path(path)) for name, path in outputs if path is not None]
    inputs = [(name, Path(path)) for name, path in inputs if path is not None]
    for i, (name, path) in enumerate(outputs):
        for other_name, other in outputs[:i] + inputs:
            if _same_file_path(path, other):
                sys.exit(f"error: {name} and {other_name} refer to the same file")
        for directory in result_dirs or ():
            root = Path(directory).resolve()
            protected = [
                root / n
                for n in ("sentences.csv", "documents.csv", "recall_pool.csv", "provenance.json", ".save_in_progress")
            ]
            checkpoint = root / "checkpoint"
            checkpoint_alias = (
                path.exists()
                and checkpoint.is_dir()
                and any(_same_file_path(path, f) for f in checkpoint.rglob("*") if f.is_file())
            )
            if (
                any(_same_file_path(path, f) for f in protected)
                or path.resolve().is_relative_to(checkpoint)
                or checkpoint_alias
            ):
                sys.exit(f"error: {name} would overwrite a result or checkpoint file")


def _write_report(out, markdown: str, report: dict, *, default, inputs=(), result_dirs=()) -> None:
    target = Path(out)
    json_target = target.with_suffix(".json")
    _check_output_paths([("report", target), ("JSON report", json_target)], inputs=inputs, result_dirs=result_dirs)
    target.write_text(markdown, encoding="utf-8")
    json_target.write_text(json.dumps(report, indent=2, default=default), encoding="utf-8")


def _parse_value(text: str):
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _key_values(items: list[str] | None, flag: str) -> dict:
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"error: {flag} expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip().replace("-", "_")] = _parse_value(v.strip())
    return out


# ------------------------------------------------------------------ sources
def _add_source_args(p: argparse.ArgumentParser) -> None:
    from .sources import REGISTRY

    g = p.add_argument_group("source")
    g.add_argument("--source", choices=sorted(REGISTRY), required=True, help="document source")
    g.add_argument("--ticker", nargs="*", default=[], help="[edgar] ticker symbols")
    g.add_argument("--cik", nargs="*", default=[], help="[edgar] CIK numbers")
    g.add_argument("--forms", nargs="*", default=["10-K"], help="[edgar] 10-K, 20-F and/or 40-F (default 10-K)")
    g.add_argument(
        "--exhibits",
        choices=["auto", "none", "all"],
        default="auto",
        help="[edgar] add Exhibit 13 (10-K) and AIF/MD&A exhibits (40-F); default auto",
    )
    g.add_argument(
        "--fiscal-year-rule",
        choices=["period_end_year", "jan_may_previous"],
        default="jan_may_previous",
        help="[edgar] fiscal-year label convention",
    )
    g.add_argument("--user-agent", default=None, help="[edgar] 'Name email@domain' as required by the SEC")
    g.add_argument("--country", nargs="*", default=[], help="[xbrl] ISO country codes, e.g. DK SE GB")
    g.add_argument("--lei", nargs="*", default=[], help="[xbrl] legal entity identifiers")
    g.add_argument("--max-pages", type=int, default=None, help="[xbrl] limit API pages (testing)")
    g.add_argument("--dir", default=None, help="[local] folder with reports")
    g.add_argument("--metadata", default=None, help="[local] metadata CSV")
    g.add_argument("--start", type=int, default=None, help="first fiscal year")
    g.add_argument("--end", type=int, default=None, help="last fiscal year")
    g.add_argument("--cache-dir", default=".aidisclose_cache", help="cache for downloads (default .aidisclose_cache)")
    g.add_argument(
        "--source-option",
        action="append",
        metavar="KEY=VALUE",
        help="[custom sources] constructor argument (repeatable)",
    )
    g.add_argument(
        "--query",
        action="append",
        metavar="KEY=VALUE",
        help="[custom sources] extra argument for list_filings (repeatable)",
    )


def _make_source(a: argparse.Namespace):
    from .sources import BUILTIN_SOURCES, REGISTRY, EdgarSource, LocalSource, XbrlFilingsSource

    cache = Path(a.cache_dir) / a.source if a.cache_dir else None
    if a.source == "edgar":
        if not a.user_agent:
            sys.exit("error: --user-agent 'Your Name your.email@domain' is required by the SEC")
        return EdgarSource(
            a.user_agent,
            cache_dir=cache,
            exhibits=getattr(a, "exhibits", "auto"),
            fiscal_year_rule=getattr(a, "fiscal_year_rule", "jan_may_previous"),
        )
    if a.source in ("xbrl", "esef"):
        return XbrlFilingsSource(cache_dir=cache)
    if a.source == "local":
        if not a.dir:
            sys.exit("error: --dir is required for the local source")
        return LocalSource(a.dir, metadata=a.metadata)
    assert a.source not in BUILTIN_SOURCES
    return REGISTRY[a.source](**_key_values(a.source_option, "--source-option"))


def _list(a: argparse.Namespace, src):
    if a.source == "edgar":
        return src.list_filings(ciks=a.cik, tickers=a.ticker, forms=a.forms, start_year=a.start, end_year=a.end)
    if a.source in ("xbrl", "esef"):
        languages = getattr(a, "languages", None) or None
        all_languages = bool(getattr(a, "all_languages", False))
        return src.list_filings(
            countries=a.country or None,
            leis=a.lei or None,
            start_year=a.start,
            end_year=a.end,
            max_pages=a.max_pages,
            english_only=not (all_languages or bool(languages and any(lang.lower() != "en" for lang in languages))),
            languages=languages,
        )
    if a.source == "local":
        filings = src.list_filings()
    else:
        filings = src.list_filings(start_year=a.start, end_year=a.end, **_key_values(a.query, "--query"))
    return [
        f
        for f in filings
        if (a.start is None or (f.fiscal_year or 0) >= a.start) and (a.end is None or (f.fiscal_year or 9999) <= a.end)
    ]


def _filings_to_frame(filings) -> pd.DataFrame:
    return pd.DataFrame([f.to_dict() for f in filings])


def _filings_from_frame(df: pd.DataFrame):
    from .models import Filing

    return [Filing.from_dict(r) for r in df.to_dict(orient="records")]


def _load_results(paths: list[str]):
    from .pipeline import Results

    parts = [Results.load(p) for p in paths]
    return parts[0] if len(parts) == 1 else Results.concat(parts)


# ----------------------------------------------------------------- backend
def _add_backend_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("classification")
    g.add_argument("--backend", choices=["rules", "supervised", "llm", "transformer"], default="rules")
    g.add_argument("--model-path", default=None, help="[supervised, transformer] trained model file or directory")
    g.add_argument("--llm-model", default=None, help="[llm] model name served by the endpoint")
    g.add_argument("--llm-model-revision", default=None, help="[llm] immutable model revision or weights digest")
    g.add_argument("--llm-base-url", default=None, help="[llm] OpenAI-compatible base URL incl. /v1")
    g.add_argument("--llm-votes", type=int, default=1, help="[llm] sampled answers per sentence")
    g.add_argument("--llm-max-tokens", type=int, default=200, help="[llm] maximum generated tokens per answer")
    g.add_argument("--llm-temperature", type=float, default=0.0)
    g.add_argument("--llm-prompt", default="codebook-v1.1", help="[llm] prompt version (codebook-v1.1, codebook-v1)")
    g.add_argument(
        "--llm-response-format",
        default="auto",
        choices=["auto", "json_schema", "json_object", "none"],
        help="[llm] structured-output mode (auto falls back when the server rejects one)",
    )
    g.add_argument("--llm-concurrency", type=int, default=1, help="[llm] parallel requests")
    g.add_argument("--lexicon", default="default", help="built-in lexicon name or YAML path")
    g.add_argument("--tiers", nargs="*", default=["core"], help="lexicon tiers (core, extended)")
    g.add_argument(
        "--sections",
        nargs="*",
        default=None,
        help="section labels and/or presets to process, e.g. 'core' or 10-K:1 10-K:1A 10-K:7 (default: all)",
    )
    g.add_argument("--all-languages", action="store_true", help="do not skip non-English documents")
    g.add_argument(
        "--languages", nargs="*", default=None, help="languages to process (default en); others need --lexicon-for"
    )
    g.add_argument(
        "--lexicon-for",
        action="append",
        metavar="LANG=PATH",
        help="lexicon YAML for a language other than English (repeatable)",
    )
    g.add_argument(
        "--recall-random", type=int, default=0, help="recall pool: random non-retrieved sentences per document"
    )
    g.add_argument("--recall-probe-cap", type=int, default=0, help="recall pool: probe-term sentences per document")


def _make_pipeline(a: argparse.Namespace):
    from .pipeline import Pipeline

    opts = {}
    if a.backend in ("supervised", "transformer"):
        if not a.model_path:
            sys.exit(f"error: --model-path is required for the {a.backend} backend")
        opts["model_path"] = a.model_path
    if a.backend == "llm":
        opts.update(
            model=a.llm_model,
            model_revision=a.llm_model_revision,
            base_url=a.llm_base_url,
            n_votes=a.llm_votes,
            max_tokens=a.llm_max_tokens,
            temperature=a.llm_temperature,
            prompt=a.llm_prompt,
            response_format=a.llm_response_format,
            concurrency=a.llm_concurrency,
            cache_path=str(Path(getattr(a, "cache_dir", None) or ".aidisclose_cache") / "llm.sqlite"),
        )
    lexicons = {}
    for item in getattr(a, "lexicon_for", None) or []:
        if "=" not in item:
            sys.exit(f"error: --lexicon-for expects LANG=PATH, got {item!r}")
        lang, path = item.split("=", 1)
        lexicons[lang.strip().lower()] = path.strip()
    try:
        return Pipeline(
            backend=a.backend,
            lexicon=a.lexicon,
            tiers=a.tiers or None,
            sections=a.sections,
            english_only=not a.all_languages,
            languages=getattr(a, "languages", None) or None,
            lexicons=lexicons or None,
            backend_options=opts,
            recall_random=getattr(a, "recall_random", 0),
            recall_probe_cap=getattr(a, "recall_probe_cap", 0),
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")


# ---------------------------------------------------------------- commands
def cmd_list(a):
    if a.source in ("xbrl", "esef"):
        _warn("XBRL listings may include interim reports; review annual status before firm-year aggregation")
    try:
        src = _make_source(a)
        filings = _list(a, src)
        _check_output_paths(
            [("filing list", a.out)],
            inputs=[("metadata", a.metadata)]
            + [("local report", f.extra.get("path")) for f in filings if f.source == "local"],
        )
        df = _filings_to_frame(filings)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    df.to_csv(a.out, index=False)
    print(f"{len(df)} filings written to {a.out}")


def cmd_run(a):
    if a.source in ("xbrl", "esef"):
        _warn("XBRL listings may include interim reports; review annual status before firm-year aggregation")
    out_dir = Path(a.out)
    source_inputs = [("metadata", a.metadata), ("filings list", a.filings)]
    result_names = ("sentences.csv", "documents.csv", "recall_pool.csv", "provenance.json", ".save_in_progress")
    _check_output_paths([("result " + n, out_dir / n) for n in result_names], inputs=source_inputs)
    if not a.no_checkpoint:
        for name, path in source_inputs:
            if path is not None and Path(path).resolve().is_relative_to((out_dir / "checkpoint").resolve()):
                sys.exit(f"error: {name} is inside the checkpoint directory")
    try:
        src = _make_source(a)
        if a.filings:
            filings = _filings_from_frame(pd.read_csv(a.filings, dtype=str, keep_default_na=False))
        else:
            filings = _list(a, src)
        local_inputs = [
            ("local report", src.input_content_path(f)) for f in filings if a.source == "local" and f.source == "local"
        ]
        _check_output_paths([("result " + n, out_dir / n) for n in result_names], inputs=local_inputs)
        if not a.no_checkpoint:
            for name, path in local_inputs:
                if path.resolve().is_relative_to((out_dir / "checkpoint").resolve()):
                    sys.exit(f"error: {name} is inside the checkpoint directory")
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if a.limit:
        filings = filings[: a.limit]
    pipe = _make_pipeline(a)
    checkpoint = Path(a.out) / "checkpoint"
    try:
        res = pipe.run(
            filings,
            source=src,
            progress=not a.quiet,
            workers=a.workers,
            executor=a.executor,
            checkpoint_dir=None if a.no_checkpoint else checkpoint,
            resume=a.resume,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    out = res.save(a.out)
    docs = res.documents
    ok = (docs["status"] == "ok").sum()
    print(f"processed {ok}/{len(docs)} documents, {len(res.sentences)} keyword sentences -> {out}")
    n_err = int(pd.to_numeric(docs.get("n_classifier_errors", 0), errors="coerce").fillna(0).sum())
    if n_err:
        _warn(
            f"{n_err} sentences could not be classified by the backend; they are listed with "
            "classification_status=error and are excluded from adoption indicators until re-run or reviewed"
        )
    failed = docs[docs["status"] == "error"]
    if len(failed):
        _warn(f"{len(failed)} documents failed; see the 'error' column of documents.csv")


def cmd_relabel(a):
    if any(_same_file_path(a.out, source) for source in a.results):
        sys.exit("error: relabel output directory must differ from every input result directory")
    _check_output_paths(
        [
            ("relabel result " + n, Path(a.out) / n)
            for n in ("sentences.csv", "documents.csv", "recall_pool.csv", "provenance.json", ".save_in_progress")
        ],
        result_dirs=a.results,
    )
    res = _load_results(a.results)
    pipe = _make_pipeline(a)
    new = pipe.relabel(res)
    new.save(a.out)
    changed = (new.sentences["state"] != res.sentences["state"]).sum()
    print(f"relabelled {len(new.sentences)} sentences with {a.backend}: {changed} changed state -> {a.out}")


def cmd_llm_estimate(a):
    from .classify.llm import LLMClassifier

    res = _load_results(a.results)
    pipe = _make_pipeline(a)
    if not isinstance(pipe.classifier, LLMClassifier):
        sys.exit("error: llm-estimate needs --backend llm")
    est = pipe.classifier.estimate(pipe.candidates_from_results(res))
    print(json.dumps(est, indent=2))


def cmd_panel(a):
    from .aggregate import firm_year_panel, group_panel, read_industry_map

    _check_output_paths(
        [("panel", a.out)],
        inputs=[("industry map", a.industry_map), ("firm attributes", a.firm_attributes)],
        result_dirs=a.results,
    )
    res = _load_results(a.results)
    industry_map = read_industry_map(a.industry_map) if a.industry_map else None
    attrs = pd.read_csv(a.firm_attributes, dtype={"firm_id": str}, keep_default_na=False) if a.firm_attributes else None
    try:
        panel = firm_year_panel(
            res,
            industry_map=industry_map,
            firm_attributes=attrs,
            repeat_threshold=a.repeat_threshold,
            unknown_years="exclude" if a.exclude_unknown_years else "error",
            xbrl_periods="reviewed" if a.confirm_xbrl_periods else "error",
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if a.level == "firm-year":
        out = panel
    else:
        by = {
            "country-year": ["country", "fiscal_year"],
            "industry-year": ["nace_section", "fiscal_year"],
            "country-industry-year": ["country", "nace_section", "fiscal_year"],
            "year": ["fiscal_year"],
        }[a.level]
        try:
            out = group_panel(panel, by=by, min_firms=a.min_firms, weight=a.weight)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows written to {a.out}")
    if not a.quiet:
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(out.head(20).to_string(index=False))


def cmd_sample(a):
    from .validate.sampling import blind, sample_for_annotation

    res = _load_results(a.results)
    n_errors = int(
        (res.sentences.get("classification_status", pd.Series("ok", index=res.sentences.index)) != "ok").sum()
    )
    if n_errors:
        _warn(f"excluded {n_errors} sentences whose classifier result is missing")
    sample = sample_for_annotation(
        res.sentences, n=a.n, strata=a.strata, min_per_stratum=a.min_per_stratum, seed=a.seed
    )
    to_annotate, key = blind(sample)
    key["dataset_sha256"] = res.dataset_digest()
    key["classification_complete"] = "true" if n_errors == 0 else "false"
    _write_sample(a, to_annotate, key, len(sample))


def _write_sample(a, to_annotate: pd.DataFrame, key: pd.DataFrame, n: int) -> None:
    from .validate.annotation import LABEL_STUDIO_CONFIG, to_label_studio, write_annotation_file
    from .validate.sampling import seal_sample_key

    key_path = a.key or str(Path(a.out).with_name(Path(a.out).stem + "_key.csv"))
    outputs = [("annotation sheet", a.out), ("sample key", key_path)]
    if getattr(a, "label_studio", None):
        ls = Path(a.label_studio)
        outputs.extend([("Label Studio tasks", ls), ("Label Studio config", ls.with_name(ls.stem + "_config.xml"))])
    _check_output_paths(outputs, result_dirs=a.results)
    key = seal_sample_key(key)
    write_annotation_file(to_annotate, a.out)
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    print(f"{n} sentences to annotate -> {a.out} (predictions and weights in {key_path})")
    if getattr(a, "label_studio", None):
        ls = Path(a.label_studio)
        ls.write_text(json.dumps(to_label_studio(to_annotate), indent=1, ensure_ascii=False), encoding="utf-8")
        cfg = ls.with_name(ls.stem + "_config.xml")
        cfg.write_text(LABEL_STUDIO_CONFIG, encoding="utf-8")
        print(f"Label Studio tasks -> {ls} (labelling interface: {cfg})")


def cmd_sample_recall(a):
    from .validate.recall import sample_recall, validate_recall_coverage

    res = _load_results(a.results)
    try:
        if "classification_status" in res.sentences and (res.sentences["classification_status"] != "ok").any():
            raise ValueError("recall estimation requires complete classification of retrieved sentences")
        validate_recall_coverage(res)
        to_annotate, key = sample_recall(res.recall_pool, n=a.n, share_probe=a.share_probe, seed=a.seed)
        key["coverage_verified"] = "true"
        key["dataset_sha256"] = res.dataset_digest()
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    _write_sample(a, to_annotate, key, len(to_annotate))


def cmd_evaluate_recall(a):
    from .validate.annotation import read_annotation_file
    from .validate.recall import estimate_recall, found_from_predictions, merge_recall_key, recall_to_markdown

    key = pd.read_csv(a.key, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    from .validate.sampling import verify_sample_key

    try:
        verify_sample_key(key)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if "coverage_verified" not in key or not key["coverage_verified"].astype(str).str.lower().eq("true").all():
        sys.exit("error: recall key lacks verified first-stage coverage; create a new sample from complete results")
    if (
        "dataset_sha256" not in key
        or key.empty
        or key["dataset_sha256"].nunique() != 1
        or not key["dataset_sha256"].iloc[0]
    ):
        sys.exit("error: recall key lacks a unique dataset fingerprint; regenerate the sample")
    dataset_id = key["dataset_sha256"].iloc[0]
    try:
        recall_gold = read_annotation_file(a.gold)
        _warn_unverified_context(recall_gold, key)
        gold = merge_recall_key(recall_gold, key)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    main = None
    found = None
    found_claim = None
    if a.main_gold:
        if not a.main_key:
            sys.exit("error: --main-gold requires --main-key for complete sample verification")
        main_key = pd.read_csv(a.main_key, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "dataset_sha256" not in main_key or main_key.empty or not main_key["dataset_sha256"].eq(dataset_id).all():
            sys.exit("error: main and recall samples come from different datasets")
        if (
            "classification_complete" not in main_key
            or not main_key["classification_complete"].str.lower().eq("true").all()
        ):
            sys.exit("error: main sample does not certify complete classification")
        main = _attach_key(read_annotation_file(a.main_gold), a.main_key)
    elif a.results:
        observed = _load_results(a.results)
        if observed.dataset_digest() != dataset_id:
            sys.exit("error: recall sample and prediction results come from different datasets")
        if (
            "classification_status" in observed.sentences
            and (observed.sentences["classification_status"] != "ok").any()
        ):
            sys.exit("error: predicted recall counts require complete classifier results")
        found = found_from_predictions(observed.sentences)
        found_claim = found_from_predictions(observed.sentences, states=("SUBSTANTIVE", "WEAK"))
        _warn("no --main-gold: retrieved SUBSTANTIVE and WEAK totals are taken from predictions (unvalidated)")
    else:
        sys.exit("error: give --main-gold/--main-key (annotated precision sample) or --results")
    try:
        rep = estimate_recall(
            gold,
            main_gold=main,
            found_substantive=found,
            found_claim=found_claim,
            n_boot=a.n_boot,
            found_source="unvalidated classifier predictions" if main is None else None,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    md = recall_to_markdown(rep)
    print(md)
    if a.out:
        _write_report(
            a.out,
            md,
            rep,
            default=float,
            inputs=[
                ("annotation", a.gold),
                ("sample key", a.key),
                ("main annotation", getattr(a, "main_gold", None)),
                ("main key", getattr(a, "main_key", None)),
            ],
            result_dirs=getattr(a, "results", None),
        )


def cmd_agreement(a):
    from .validate.annotation import agreement_report, agreement_to_markdown, from_label_studio, read_annotation_file

    if a.label_studio:
        tables = from_label_studio(a.label_studio)
    else:
        if not a.annotations or len(a.annotations) < 2:
            sys.exit("error: give two or more --annotations files (one per annotator) or --label-studio")
        for i, path in enumerate(a.annotations):
            if any(_same_file_path(path, previous) for previous in a.annotations[:i]):
                sys.exit("error: the same annotation file cannot represent multiple annotators")
        tables = [read_annotation_file(p) for p in a.annotations]
    try:
        rep = agreement_report(tables, column=a.column)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    if not isinstance(tables, pd.DataFrame):
        stems = [Path(p).stem for p in a.annotations]
        names = (
            stems
            if len(set(stems)) == len(stems) and "sentence_id" not in stems
            else [f"{stem} [{i + 1}]" for i, stem in enumerate(stems)]
        )
        mapping = {f"annotator_{i + 1}": name for i, name in enumerate(names)}
        rep["annotators"] = names
        for pair in rep["pairs"]:
            pair["a"], pair["b"] = mapping[pair["a"]], mapping[pair["b"]]
        rep["disagreements"] = [{mapping.get(k, k): v for k, v in row.items()} for row in rep["disagreements"]]
    md = agreement_to_markdown(rep)
    print(md)
    if a.out:
        _write_report(
            a.out,
            md,
            rep,
            default=str,
            inputs=[("annotation", p) for p in (getattr(a, "annotations", None) or [])]
            + [("Label Studio input", getattr(a, "label_studio", None))],
            result_dirs=getattr(a, "results", None),
        )


def cmd_import_label_studio(a):
    from .validate.annotation import from_label_studio

    _check_output_paths([("annotation output", a.out)], inputs=[("Label Studio export", a.export)])
    df = from_label_studio(a.export)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    n_ann = df["annotator"].nunique() if len(df) else 0
    print(f"{len(df)} annotations by {n_ann} annotators -> {a.out}")


def _load_gold(a):
    from .validate.annotation import read_annotation_file

    gold = read_annotation_file(a.gold)
    if "gold_state" not in gold.columns:
        sys.exit("error: the gold file needs a 'gold_state' column")
    return gold


def _warn_unverified_context(gold: pd.DataFrame, key: pd.DataFrame) -> None:
    if "sample_schema" not in key:
        _warn(
            "legacy sample key does not protect annotation context; regenerate the sample if context integrity is required"
        )
    visible = ("text", "firm_name", "fiscal_year", "section", "prev_text", "next_text")
    if any(c in key and c not in gold for c in visible):
        _warn("annotation file lacks some sampled text or context fields; those fields could not be verified")


def _attach_key(gold: pd.DataFrame, key_path: str | None) -> pd.DataFrame:
    """Add sampling and grouping metadata from the key file written by 'sample'."""
    if not key_path:
        return gold
    if not Path(key_path).exists():
        sys.exit(f"error: key file {key_path} not found")
    key = pd.read_csv(key_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    from .validate.sampling import merge_sample_key, verify_sample_key

    try:
        verify_sample_key(key)
        columns = [
            c
            for c in (
                "sentence_id",
                "text",
                "firm_name",
                "fiscal_year",
                "section",
                "prev_text",
                "next_text",
                "sampling_weight",
                "sampling_stratum",
                "filing_id",
                "firm_id",
            )
            if c in key
        ]
        _warn_unverified_context(gold, key)
        return merge_sample_key(gold, key[columns])
    except ValueError as exc:
        sys.exit(f"error: {exc}")


def _weights_and_strata(df: pd.DataFrame, weighted: bool):
    if not weighted:
        return None, None
    if "sampling_weight" not in df.columns:
        sys.exit("error: --weighted requires sampling weights from the sample key")
    w = pd.to_numeric(df["sampling_weight"], errors="coerce")
    if w.isna().any() or (~np.isfinite(w)).any() or (w <= 0).any():
        sys.exit("error: sampling weights must be finite and positive")
    strata = df["sampling_stratum"].tolist() if "sampling_stratum" in df.columns else None
    return w.tolist(), strata


def cmd_evaluate(a):
    from .validate.metrics import bootstrap_report, classification_report, report_to_markdown
    from .validate.sampling import merge_annotations

    if a.weighted and not a.key:
        sys.exit("error: --weighted requires --key to verify the complete annotation sample")
    if a.weighted and "sentence_id" not in a.on:
        sys.exit("error: --weighted requires sentence_id among --on join columns")
    gold = _attach_key(_load_gold(a), a.key)
    if (
        a.weighted
        and not gold["gold_state"].astype(str).str.strip().str.upper().isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"]).all()
    ):
        sys.exit("error: weighted evaluation requires a valid label for every sampled row")
    res = _load_results(a.results)
    if a.weighted:
        sample_key = pd.read_csv(a.key, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if (
            "dataset_sha256" not in sample_key
            or sample_key.empty
            or not sample_key["dataset_sha256"].eq(res.dataset_digest()).all()
        ):
            sys.exit("error: annotation sample and prediction results come from different datasets")
        if (
            "classification_complete" not in sample_key
            or not sample_key["classification_complete"].str.lower().eq("true").all()
        ):
            sys.exit(
                "error: weighted population evaluation requires complete classification; resolve errors and resample"
            )
    pred = res.sentences.copy()
    keys = a.on
    if len(keys) == 1:
        try:
            merged = merge_annotations(gold, pred.rename(columns={"text": "_pred_text"}), on=keys[0])
        except ValueError as exc:
            sys.exit(f"error: {exc}")
    else:
        overlap = {"state", "reason", "classification_status", "confidence", "backend", "details", "_pred_text"} & set(
            gold.columns
        )
        if overlap:
            sys.exit(f"error: gold data contains prediction columns: {sorted(overlap)}")
        merged = gold.merge(pred.rename(columns={"text": "_pred_text"}), on=keys, how="inner", validate="one_to_one")
        merged["gold_state"] = merged["gold_state"].astype(str).str.strip().str.upper().str.replace(" ", "_")
    if "text" in gold.columns and "_pred_text" in merged.columns and "text" in merged.columns:
        mismatch = merged["text"].astype(str).str.strip() != merged["_pred_text"].astype(str).str.strip()
        if mismatch.any():
            if a.weighted:
                sys.exit("error: weighted sample text does not match predictions")
            _warn(
                f"{int(mismatch.sum())} annotated rows match a prediction with different text "
                "(different extraction or pipeline settings?); they are excluded"
            )
            merged = merged[~mismatch]
    n_unmatched = len(gold) - len(merged)
    if n_unmatched > 0:
        if a.weighted:
            sys.exit("error: weighted sample does not match all predictions")
        _warn(f"{n_unmatched} annotated rows could not be matched to predictions")
    if "classification_status" in merged:
        failed = merged["classification_status"] != "ok"
        if failed.any():
            if a.weighted:
                sys.exit("error: weighted sample contains classifier errors")
            _warn(f"excluded {int(failed.sum())} matched rows with classifier errors")
            merged = merged[~failed]
    valid = merged["gold_state"].astype(str).str.upper().isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"])
    valid &= merged["state"].astype(str).str.upper().isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"])
    if (~valid).any():
        if a.weighted:
            sys.exit("error: weighted sample contains invalid labels")
        _warn(f"excluded {int((~valid).sum())} rows without valid gold and predicted labels")
        merged = merged[valid]
    if merged.empty:
        sys.exit("error: no annotated rows with valid predictions could be evaluated")
    w, strata = _weights_and_strata(merged, a.weighted)
    y_true, y_pred = merged["gold_state"].tolist(), merged["state"].tolist()
    rep = classification_report(y_true, y_pred, weights=w)
    cluster_col = a.cluster_by
    if cluster_col == "none":
        clusters = None
    else:
        cluster_col = next((c for c in (cluster_col, cluster_col + "_x", cluster_col + "_y") if c in merged), None)
        if cluster_col is None:
            _warn(f"cluster column {a.cluster_by!r} is unavailable; using row bootstrap")
            clusters = None
        else:
            cluster_values = merged[cluster_col].fillna("").astype(str)
            if (cluster_values.str.strip() == "").any():
                _warn(f"cluster column {cluster_col!r} has missing identifiers; using row bootstrap")
                clusters = None
            else:
                clusters = cluster_values.tolist()
    rep["ci"] = bootstrap_report(
        y_true,
        y_pred,
        weights=w,
        strata=strata if clusters is None else None,
        clusters=clusters,
        n_boot=a.n_boot,
    )
    rep["ci_method"] = "cluster" if clusters is not None else "stratified_rows" if strata is not None else "row"
    md = report_to_markdown(rep, title=a.title)
    print(md)
    if a.out:
        _write_report(
            a.out,
            md,
            rep,
            default=float,
            inputs=[
                ("annotation", a.gold),
                ("sample key", a.key),
                ("main annotation", getattr(a, "main_gold", None)),
                ("main key", getattr(a, "main_key", None)),
            ],
            result_dirs=getattr(a, "results", None),
        )


def _gold_candidates(gold: pd.DataFrame, lexicon: str = "default", tiers=("core",)):
    """Rebuild candidates from annotated rows with the same lexicon settings as the pipeline."""
    from .lexicon import Lexicon
    from .models import Candidate

    lex = Lexicon.load(lexicon, tiers=tiers)
    cands = []
    for i, r in gold.iterrows():
        hits = lex.find(str(r["text"]))
        cands.append(
            Candidate(
                filing_id=str(r.get("filing_id", "")),
                sentence_id=str(r.get("sentence_id", i)),
                section=str(r.get("section", "full") or "full"),
                text=str(r["text"]),
                prev_text=str(r.get("prev_text", "") or ""),
                next_text=str(r.get("next_text", "") or ""),
                firm_name=str(r.get("firm_name", "") or ""),
                terms=hits.term_ids("valid"),
                term_spans=[(h.start, h.end) for h in hits.valid],
                filtered=hits.term_ids("filtered"),
            )
        )
    return cands


def cmd_train(a):
    from .classify.supervised import SupervisedClassifier
    from .validate.metrics import report_to_markdown

    outputs = [("trained model", a.model_out)]
    if a.cv and a.oof_out:
        outputs.append(("out-of-fold predictions", a.oof_out))
    _check_output_paths(outputs, inputs=[("gold annotations", a.gold), ("sample key", a.key)], result_dirs=a.results)
    if a.weighted and not a.key:
        sys.exit("error: weighted training requires the complete sample key")
    if a.key and not a.results:
        sys.exit("error: training with a sample key requires --results from the original sample")
    if a.results and not a.key:
        sys.exit("error: --results for training requires the matching sample --key")
    gold = _attach_key(_load_gold(a), a.key)
    gold["gold_state"] = gold["gold_state"].astype(str).str.strip().str.upper()
    valid = gold["gold_state"].isin(["SUBSTANTIVE", "WEAK", "NO_CLAIM"])
    if a.weighted and not valid.all():
        sys.exit("error: weighted training requires a valid label for every sampled row")
    if (~valid).any():
        _warn(f"excluded {int((~valid).sum())} rows without a valid annotation")
    gold = gold.loc[valid].reset_index(drop=True)
    if a.key:
        from .pipeline import Pipeline

        key = pd.read_csv(a.key, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        res = _load_results(a.results)
        if "dataset_sha256" not in key or key.empty or not key["dataset_sha256"].eq(res.dataset_digest()).all():
            sys.exit("error: annotation sample and training results come from different datasets")
        if a.weighted and (
            "classification_complete" not in key or not key["classification_complete"].str.lower().eq("true").all()
        ):
            sys.exit("error: weighted training requires complete classification; resolve errors and resample")
        try:
            candidates = Pipeline(lexicon=a.lexicon, tiers=a.tiers or None).candidates_from_results(res)
        except ValueError as exc:
            sys.exit(f"error: {exc}")
        by_id = {c.sentence_id: c for c in candidates}
        if len(by_id) != len(candidates):
            sys.exit("error: original results have duplicate sentence identifiers")
        source_ids = set(res.sentences["sentence_id"].astype(str))
        if not set(gold["sentence_id"].astype(str)).issubset(source_ids):
            sys.exit("error: sample contains sentence identifiers absent from training results")
        eligible = gold["sentence_id"].astype(str).isin(by_id)
        if (~eligible).any():
            _warn(f"excluded {int((~eligible).sum())} baseline-only or filtered rows not sent to the classifier")
        gold = gold.loc[eligible].reset_index(drop=True)
        cands = [by_id[str(sid)] for sid in gold["sentence_id"]]
    else:
        _warn(
            "training without original results reconstructs term matches from isolated sentences; context-dependent filtering cannot be reproduced"
        )
        cands = _gold_candidates(gold, a.lexicon, a.tiers or None)
    if gold.empty:
        sys.exit("error: no classifier-eligible annotated rows remain for training")
    labels = gold["gold_state"].tolist()
    if a.cv:
        w, strata = _weights_and_strata(gold, a.weighted)
        if a.cv_group_by == "sentence":
            groups = None
            _warn("sentence-level CV can put sentences from one filing in both train and test folds")
            group_label = "sentence-level"
        else:
            group_col = a.cv_group_by
            if group_col not in gold and group_col == "firm_id" and "filing_id" in gold:
                group_col = "filing_id"
                _warn("firm_id is missing; grouped CV is using filing_id")
            if group_col not in gold:
                sys.exit(f"error: --cv-group-by {a.cv_group_by} requires that column in the gold/key data")
            groups = gold[group_col].fillna("").astype(str).tolist()
            if any(not group for group in groups):
                sys.exit(f"error: --cv-group-by {group_col} contains missing values")
            group_label = f"grouped by {group_col}"
        rep = SupervisedClassifier.cross_validate(
            cands,
            labels,
            k=a.cv,
            weights=w,
            strata=strata,
            groups=groups,
            n_boot=a.n_boot,
        )
        print(
            report_to_markdown(
                rep,
                title=f"{a.cv}-fold cross-validation {group_label} (classifier-eligible sample; out-of-fold predictions)",
            )
        )
        if a.oof_out:
            oof = gold.copy()
            oof["state"] = rep["oof_predictions"]
            oof.to_csv(a.oof_out, index=False)
            print(f"out-of-fold predictions -> {a.oof_out}")
    weights, _ = _weights_and_strata(gold, a.weighted)
    model = SupervisedClassifier.train(cands, labels, sample_weight=weights)
    model.save(a.model_out)
    print(f"model trained on {len(labels)} sentences -> {a.model_out}")


def cmd_eurostat(a):
    from .validate.external import compare_with_external, eurostat_ai_share, eurostat_ai_share_by_industry

    _check_output_paths(
        [("comparison output", a.out)], inputs=[("panel", a.panel), ("Eurostat input", a.eurostat_file)]
    )
    panel = pd.read_csv(a.panel, dtype={"nace_section": str, "country": str})
    try:
        if a.dataset == "isoc_eb_ain2":
            if a.size not in (None, "GE10"):
                raise ValueError("isoc_eb_ain2 is limited to GE10; --size must be GE10 or omitted")
            if a.nace_aggregate is not None:
                raise ValueError("--nace-aggregate applies to isoc_eb_ai, not the industry dataset")
            if "nace_section" not in panel.columns:
                sys.exit("error: isoc_eb_ain2 needs a country-industry-year panel (--level country-industry-year)")
            ext = eurostat_ai_share_by_industry(path=a.eurostat_file)
            keys_g, keys_e = ("country", "nace_section", "fiscal_year"), ("country", "nace_section", "year")
        else:
            ext = eurostat_ai_share(
                path=a.eurostat_file, size_emp=a.size or "GE250", nace_r2=a.nace_aggregate or "C10-S951_X_K"
            )
            keys_g, keys_e = ("country", "fiscal_year"), ("country", "year")
        merged, summary = compare_with_external(
            panel, ext, group_keys=keys_g, external_keys=keys_e, year_lag=a.lag, min_firms=a.min_firms
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    merged.to_csv(a.out, index=False)
    print(summary.to_string(index=False))
    print(f"merged table -> {a.out}")


def cmd_benchmark(a):
    from .validate.external import compare_with_external, load_benchmark

    _check_output_paths([("comparison output", a.out)], inputs=[("panel", a.panel), ("benchmark input", a.file)])
    rename = dict(item.split("=", 1) for item in (a.rename or []))
    percent = {"auto": None, "percent": True, "fraction": False}[a.scale]
    ext_keys = [k if k != "fiscal_year" else "year" for k in a.keys]
    try:
        ext = load_benchmark(a.file, value=a.value, keys=ext_keys, rename=rename, percent=percent)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    panel = pd.read_csv(a.panel, dtype={"nace_section": str, "country": str})
    group_keys = [k if k != "year" else "fiscal_year" for k in a.keys]
    try:
        merged, summary = compare_with_external(
            panel,
            ext,
            group_keys=group_keys,
            external_keys=ext_keys,
            external_value="benchmark_share",
            year_lag=a.lag,
            min_firms=a.min_firms,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    merged.to_csv(a.out, index=False)
    print(summary.to_string(index=False))
    print(f"merged table -> {a.out}")


def cmd_qa(a):
    from .qa import quality_report, quality_to_markdown

    rep = quality_report(_load_results(a.results), min_words=a.min_words)
    md = quality_to_markdown(rep)
    print(md)
    if a.out:
        _write_report(
            a.out,
            md,
            rep,
            default=str,
            inputs=[("annotation", p) for p in (getattr(a, "annotations", None) or [])]
            + [("Label Studio input", getattr(a, "label_studio", None))],
            result_dirs=getattr(a, "results", None),
        )
    if a.strict and rep["summary"]["n_warnings"]:
        sys.exit(1)


def cmd_doctor(a):
    from .doctor import check_environment, check_network, format_rows

    rows = check_environment(a.cache_dir)
    if a.network:
        ua = a.user_agent or os.environ.get("AIDISCLOSE_USER_AGENT")
        if not ua:
            sys.exit("error: --network needs --user-agent 'Name email@domain' (or $AIDISCLOSE_USER_AGENT)")
        rows += check_network(ua, llm_base_url=a.llm_base_url or os.environ.get("AIDISCLOSE_LLM_BASE_URL"))
    print(format_rows(rows))
    if any(r["status"] == "fail" for r in rows):
        sys.exit(1)


def cmd_lexicon(a):
    from .lexicon import Lexicon

    lex = Lexicon.load(a.name, tiers=a.tiers or None)
    print(json.dumps(lex.describe(), indent=2))
    if a.test:
        hits = lex.find(a.test)
        print("valid:", [(h.term_id, h.surface) for h in hits.valid])
        print("filtered:", [(h.term_id, h.surface, hits.reasons.get(h.term_id)) for h in hits.filtered])
    elif not a.quiet:
        for t in lex.terms:
            flags = ",".join(k for k in ("ambiguous", "requires_cooccurrence", "case_sensitive") if getattr(t, k))
            print(f"  {t.id:28s} {t.category:10s} {t.tier:9s} {flags}")


def cmd_demo(a):
    from .aggregate import firm_year_panel, group_panel
    from .pipeline import Pipeline
    from .sources import LocalSource
    from .validate.metrics import classification_report, report_to_markdown

    data = resources.files("aidisclose.demo").joinpath("data")
    with resources.as_file(data) as ddir:
        src = LocalSource(ddir, metadata=Path(ddir) / "metadata.csv")
        res = Pipeline().run(src.list_filings(), source=src)
        gold = pd.read_csv(Path(ddir) / "demo_gold.csv", dtype={"fiscal_year": "Int64"})
    out = Path(a.out)
    res.save(out / "results")
    panel = firm_year_panel(res)
    panel.to_csv(out / "firm_year_panel.csv", index=False)
    country = group_panel(panel, by=["country", "fiscal_year"])
    country.to_csv(out / "country_year_panel.csv", index=False)
    cols = [
        "firm_name",
        "fiscal_year",
        "country",
        "n_baseline_sentences",
        "n_substantive",
        "n_weak",
        "keyword_any",
        "substantive_any",
    ]
    print("SYNTHETIC DEMO CORPUS (fictional companies) - firm-year indicators\n")
    with pd.option_context("display.width", 200):
        print(panel[cols].to_string(index=False))
        skipped = res.documents[res.documents["status"] != "ok"][["firm_name", "status"]]
        if len(skipped):
            print("\nskipped documents:\n" + skipped.to_string(index=False))
        yearly = group_panel(panel, by=["fiscal_year"])
        print("\nprevalence by year (share of firms):")
        print(
            yearly[
                [
                    "fiscal_year",
                    "n_firms",
                    "share_keyword",
                    "share_claim",
                    "share_substantive",
                    "gap_keyword_substantive_pp",
                ]
            ].to_string(index=False)
        )
        ev = res.evidence(firm_id="northwind", fiscal_year=2023)
        print("\nevidence for Northwind Logistics, FY2023:")
        for _, r in ev.iterrows():
            print(f"  [{r['reason']}] {r['text']}")
    m = gold.merge(res.sentences, on=["firm_id", "fiscal_year", "text"], how="inner")
    rep = classification_report(m["gold_state"].tolist(), m["state"].tolist())
    print(
        "\nAgreement with the labels the demo sentences were written to illustrate "
        "(a functional check, NOT a validation):"
    )
    print(f"  {len(m)} sentences, accuracy {rep['accuracy']:.3f}, macro-F1 {rep['macro_f1']:.3f}")
    print(f"\noutputs written to {out.resolve()}")
    (out / "demo_agreement.md").write_text(
        report_to_markdown(rep, "Demo corpus agreement (synthetic)"), encoding="utf-8"
    )


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aidisclose", description="Measure substantive AI adoption in corporate disclosures."
    )
    p.add_argument("--version", action="version", version=f"aidisclose {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument(
        "--plugin", action="append", metavar="MODULE", help="import a module that registers custom sources (repeatable)"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("demo", help="run the pipeline on the bundled synthetic corpus (offline)")
    s.add_argument("--out", default="aidisclose-demo")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("list", help="list filings from a source")
    _add_source_args(s)
    s.add_argument("--all-languages", action="store_true", help="[xbrl] include non-English language hints")
    s.add_argument("--out", required=True, help="output CSV")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("run", help="fetch filings and classify AI sentences")
    _add_source_args(s)
    _add_backend_args(s)
    s.add_argument("--filings", default=None, help="CSV produced by 'aidisclose list' (otherwise list on the fly)")
    s.add_argument("--limit", type=int, default=None, help="process only the first N filings")
    s.add_argument("--out", required=True, help="output directory")
    s.add_argument("--workers", type=int, default=1, help="concurrent workers (default 1)")
    s.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="thread",
        help="threads for downloads and LLM calls, processes for CPU-bound local files",
    )
    s.add_argument("--resume", action="store_true", help="continue an interrupted run in the same --out directory")
    s.add_argument("--no-checkpoint", action="store_true", help="do not write per-filing checkpoints")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("relabel", help="re-classify the sentences of a results directory with another backend")
    s.add_argument("--results", required=True, nargs="+", help="one or more results directories")
    _add_backend_args(s)
    s.add_argument("--cache-dir", default=".aidisclose_cache", help="cache for LLM responses")
    s.add_argument("--out", required=True, help="output directory")
    s.set_defaults(func=cmd_relabel)

    s = sub.add_parser("llm-estimate", help="count LLM requests and tokens needed to label a results directory")
    s.add_argument("--results", required=True, nargs="+", help="one or more results directories")
    _add_backend_args(s)
    s.add_argument("--cache-dir", default=".aidisclose_cache", help="cache for LLM responses")
    s.set_defaults(func=cmd_llm_estimate, backend="llm")

    s = sub.add_parser("panel", help="aggregate results into firm-year or group-year indicators")
    s.add_argument("--results", required=True, nargs="+", help="one or more results directories")
    s.add_argument(
        "--level",
        default="firm-year",
        choices=["firm-year", "country-year", "industry-year", "country-industry-year", "year"],
    )
    s.add_argument("--min-firms", type=int, default=1)
    s.add_argument("--industry-map", default=None, help="CSV with industry_code,group to override NACE sections")
    s.add_argument("--firm-attributes", default=None, help="CSV with firm_id [,fiscal_year] and extra columns")
    s.add_argument("--weight", default=None, help="column for weighted group means (e.g. employees)")
    s.add_argument("--repeat-threshold", type=float, default=0.8, help="similarity above which a claim repeats")
    s.add_argument(
        "--exclude-unknown-years",
        action="store_true",
        help="exclude processed reports without a fiscal year (with warning)",
    )
    s.add_argument(
        "--confirm-xbrl-periods",
        action="store_true",
        help="confirm included XBRL reports were manually verified as annual",
    )
    s.add_argument("--out", required=True)
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_panel)

    s = sub.add_parser("sample", help="draw a stratified sample for human annotation")
    s.add_argument("--results", required=True, nargs="+", help="one or more results directories")
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--strata", nargs="*", default=["state"])
    s.add_argument("--min-per-stratum", type=int, default=20)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True, help="annotation sheet without predictions (.xlsx or .csv)")
    s.add_argument("--key", default=None, help="CSV with predictions and sampling weights")
    s.add_argument("--label-studio", default=None, help="also write Label Studio tasks (JSON) and config")
    s.set_defaults(func=cmd_sample)

    s = sub.add_parser("sample-recall", help="draw non-retrieved sentences from the recall pool for annotation")
    s.add_argument("--results", required=True, nargs="+", help="results directories with recall_pool.csv")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--share-probe", type=float, default=0.5, help="share of the sample from the probe stratum")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--out", required=True, help="annotation sheet (.xlsx or .csv)")
    s.add_argument("--key", default=None, help="CSV with recall weights")
    s.add_argument("--label-studio", default=None, help="also write Label Studio tasks (JSON) and config")
    s.set_defaults(func=cmd_sample_recall)

    s = sub.add_parser("evaluate-recall", help="estimate missed claims and lexicon recall")
    s.add_argument("--gold", required=True, help="annotated recall sample")
    s.add_argument("--key", required=True, help="key file from 'sample-recall'")
    s.add_argument("--main-gold", default=None, help="annotated precision sample (from 'sample')")
    s.add_argument("--main-key", default=None, help="key file from 'sample'")
    s.add_argument("--results", nargs="*", default=None, help="fallback: use predicted SUBSTANTIVE counts")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--out", default=None, help="Markdown report path (a .json copy is written too)")
    s.set_defaults(func=cmd_evaluate_recall)

    s = sub.add_parser("agreement", help="inter-annotator agreement (kappa, alpha)")
    s.add_argument("--annotations", nargs="*", default=None, help="one annotated file per annotator")
    s.add_argument("--label-studio", default=None, help="Label Studio JSON export with several annotators")
    s.add_argument("--column", default="gold_state", help="label column (default gold_state)")
    s.add_argument("--out", default=None, help="Markdown report path (a .json copy is written too)")
    s.set_defaults(func=cmd_agreement)

    s = sub.add_parser("import-label-studio", help="convert a Label Studio JSON export to an annotation CSV")
    s.add_argument("--export", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_import_label_studio)

    s = sub.add_parser("evaluate", help="compare annotations with predictions")
    s.add_argument("--gold", required=True, help="annotated CSV with gold_state")
    s.add_argument("--results", required=True, nargs="+", help="one or more results directories")
    s.add_argument("--key", default=None, help="key file from 'sample' (sampling weights and strata)")
    s.add_argument(
        "--on",
        nargs="+",
        default=["sentence_id"],
        help="join columns (default sentence_id; required for weighted evaluation)",
    )
    s.add_argument("--weighted", action="store_true", help="use sampling weights (population estimates)")
    s.add_argument(
        "--cluster-by",
        choices=["filing_id", "firm_id", "none"],
        default="filing_id",
        help="bootstrap whole filings/firms for confidence intervals (default filing_id)",
    )
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--title", default="Validation report")
    s.add_argument("--out", default=None, help="Markdown report path (a .json copy is written too)")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("train", help="train the supervised backend on annotations")
    s.add_argument("--gold", required=True)
    s.add_argument("--model-out", required=True)
    s.add_argument("--cv", type=int, default=0, help="also report k-fold cross-validation")
    s.add_argument(
        "--cv-group-by",
        choices=["firm_id", "filing_id", "sentence"],
        default="firm_id",
        help="keep firms or filings within one CV fold; sentence reproduces random-sentence CV",
    )
    s.add_argument("--key", default=None, help="key file from 'sample'; requires matching --results")
    s.add_argument("--results", nargs="+", default=None, help="original results used to produce the sample")
    s.add_argument("--weighted", action="store_true", help="use sampling weights for training and cross-validation")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--oof-out", default=None, help="CSV with out-of-fold predictions")
    s.add_argument("--lexicon", default="default", help="lexicon used by the pipeline")
    s.add_argument("--tiers", nargs="*", default=["core"], help="lexicon tiers used by the pipeline")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("eurostat", help="correlate a country-year panel with Eurostat AI-use shares")
    s.add_argument("--panel", required=True, help="country-year panel CSV ('aidisclose panel --level country-year')")
    s.add_argument("--eurostat-file", required=True, help="JSON-stat file of the dataset")
    s.add_argument(
        "--dataset",
        default="isoc_eb_ai",
        choices=["isoc_eb_ai", "isoc_eb_ain2"],
        help="isoc_eb_ai: country x size class; isoc_eb_ain2: country x NACE section (10+ employees)",
    )
    s.add_argument("--size", default=None, help="enterprise size: country default GE250; industry only GE10")
    s.add_argument(
        "--nace-aggregate", default=None, help="[isoc_eb_ai] exact NACE activity aggregate (default C10-S951_X_K)"
    )
    s.add_argument("--lag", type=int, default=0, help="survey year minus fiscal year")
    s.add_argument("--min-firms", type=int, default=5)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_eurostat)

    s = sub.add_parser("benchmark", help="correlate a group panel with any external benchmark (CSV)")
    s.add_argument("--panel", required=True, help="group panel CSV from 'aidisclose panel'")
    s.add_argument("--file", required=True, help="benchmark CSV")
    s.add_argument("--value", required=True, help="benchmark column with the share or percentage")
    s.add_argument("--keys", nargs="+", default=["country", "year"], help="join keys (year = fiscal year)")
    s.add_argument("--rename", nargs="*", default=None, metavar="FILE_COL=KEY", help="rename benchmark columns")
    s.add_argument("--scale", choices=["auto", "percent", "fraction"], default="auto")
    s.add_argument("--lag", type=int, default=0, help="benchmark year minus fiscal year")
    s.add_argument("--min-firms", type=int, default=5)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_benchmark)

    s = sub.add_parser("qa", help="quality checks for a results directory")
    s.add_argument("--results", required=True, nargs="+")
    s.add_argument("--min-words", type=int, default=2000)
    s.add_argument("--out", default=None, help="Markdown report path (a .json copy is written too)")
    s.add_argument("--strict", action="store_true", help="exit with status 1 if there are warnings")
    s.set_defaults(func=cmd_qa)

    s = sub.add_parser("doctor", help="check the installation and (optionally) access to the data sources")
    s.add_argument("--network", action="store_true", help="also test SEC, filings.xbrl.org and Eurostat")
    s.add_argument("--user-agent", default=None, help="'Name email@domain' for the network checks")
    s.add_argument("--llm-base-url", default=None, help="also test an OpenAI-compatible endpoint")
    s.add_argument("--cache-dir", default=".aidisclose_cache")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("lexicon", help="inspect a lexicon or test a sentence")
    s.add_argument("--name", default="default")
    s.add_argument("--tiers", nargs="*", default=["core"])
    s.add_argument("--test", default=None, help="sentence to match")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(func=cmd_lexicon)
    return p


def _load_plugins(argv: list[str]) -> None:
    """Import --plugin modules and entry-point adapters before the parser is built."""
    from .sources import load_entry_point_sources

    load_entry_point_sources()
    for i, arg in enumerate(argv):
        mod = None
        if arg == "--plugin" and i + 1 < len(argv):
            mod = argv[i + 1]
        elif arg.startswith("--plugin="):
            mod = arg.split("=", 1)[1]
        if mod:
            sys.path.insert(0, str(Path.cwd()))
            importlib.import_module(mod)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _load_plugins(argv)
    parser = build_parser()
    a = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(a, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    a.func(a)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
