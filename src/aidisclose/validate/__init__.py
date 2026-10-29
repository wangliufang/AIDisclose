"""Validation kit: annotation sampling and files, accuracy and agreement metrics,
recall audit and external benchmarks."""

from .annotation import (
    LABEL_STUDIO_CONFIG,
    agreement_report,
    agreement_to_markdown,
    fleiss_kappa,
    from_label_studio,
    krippendorff_alpha_nominal,
    read_annotation_file,
    to_label_studio,
    write_annotation_file,
)
from .external import (
    compare_with_external,
    correlation,
    eurostat_ai_share,
    eurostat_ai_share_by_industry,
    load_benchmark,
    load_eurostat,
    parse_jsonstat,
)
from .metrics import (
    bootstrap_ci,
    bootstrap_report,
    classification_report,
    cohen_kappa,
    confusion_matrix,
    report_to_markdown,
)
from .recall import estimate_recall, recall_to_markdown, sample_recall
from .sampling import blind, merge_annotations, sample_for_annotation

__all__ = [
    "LABEL_STUDIO_CONFIG",
    "agreement_report",
    "agreement_to_markdown",
    "fleiss_kappa",
    "from_label_studio",
    "krippendorff_alpha_nominal",
    "read_annotation_file",
    "to_label_studio",
    "write_annotation_file",
    "compare_with_external",
    "correlation",
    "eurostat_ai_share",
    "eurostat_ai_share_by_industry",
    "load_benchmark",
    "load_eurostat",
    "parse_jsonstat",
    "bootstrap_ci",
    "bootstrap_report",
    "classification_report",
    "cohen_kappa",
    "confusion_matrix",
    "report_to_markdown",
    "estimate_recall",
    "recall_to_markdown",
    "sample_recall",
    "blind",
    "merge_annotations",
    "sample_for_annotation",
]
