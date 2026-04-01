from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .annotation_parsing import count_invalid_boxes, filter_annotation_contents
from .constants import DEFAULT_DATASET_ROOT, LABELS


def _normalize_image_path(parser_path: str) -> str:
    path = Path(parser_path)
    parts = list(path.parts)
    if "Manga109" in parts:
        start = parts.index("Manga109")
        rel = Path(*parts[start:])
        return f"./{rel.as_posix()}"
    return f"./{path.as_posix()}"


def generate_metadata(dataset_root: Path, include_empty: bool = True, fix_invalid_boxes: bool = False) -> pd.DataFrame:
    try:
        import manga109api
    except ImportError as exc:  # pragma: no cover - exercised only in real runs
        raise RuntimeError(
            "manga109api is required for metadata generation. Install it before running this command."
        ) from exc

    parser = manga109api.Parser(root_dir=str(dataset_root))
    rows = []
    for book in parser.books:
        annotation = parser.get_annotation(book=book, separate_by_tag=False)
        for page_idx, page_annotation in enumerate(annotation["page"]):
            image_path = _normalize_image_path(parser.img_path(book=book, index=page_idx))
            cleaned_annotation = (
                filter_annotation_contents(page_annotation, allowed_labels=LABELS) if fix_invalid_boxes else page_annotation
            )
            if not include_empty and not cleaned_annotation.get("contents"):
                continue
            rows.append(
                {
                    "image_path": image_path,
                    "book": book,
                    "page": page_idx,
                    "image_annotation": cleaned_annotation,
                }
            )
    return pd.DataFrame.from_records(rows)


def write_default_metadata(dataset_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    full_df = generate_metadata(dataset_root, include_empty=True, fix_invalid_boxes=False)
    condensed_df = generate_metadata(dataset_root, include_empty=False, fix_invalid_boxes=False)
    condensed_fixed_df = generate_metadata(dataset_root, include_empty=False, fix_invalid_boxes=True)

    outputs = {
        "data_full.pkl": full_df,
        "data_condensed.pkl": condensed_df,
        "data_condensed_fixed.pkl": condensed_fixed_df,
    }
    for filename, df in outputs.items():
        df.to_pickle(output_dir / filename)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Manga109 metadata pickle files.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    outputs = write_default_metadata(args.dataset_root, args.output_dir)
    print(f"Wrote metadata files to {args.output_dir.resolve()}")
    for filename, df in outputs.items():
        invalid_boxes = sum(count_invalid_boxes(annotation) for annotation in df["image_annotation"])
        print(f"- {filename}: {len(df)} pages, invalid boxes remaining={invalid_boxes}")


if __name__ == "__main__":  # pragma: no cover
    main()
