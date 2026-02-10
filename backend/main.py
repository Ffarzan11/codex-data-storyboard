from __future__ import annotations

from io import StringIO
from itertools import combinations
from math import sqrt
import json
import os
import re
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="CSV Storyboard API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ANALYSIS_CACHE: dict[str, dict[str, str]] = {}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB


def _load_env_file() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("\"").strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        # If .env can't be read, continue with system env vars.
        return


_load_env_file()


def _safe_number(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isinf(number):
        return None
    return number


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _humanize_field_name(field: str) -> str:
    cleaned = field.replace("_", " ").strip()
    return re.sub(r"\s+", " ", cleaned).title()


def _field_plain_meaning(field: str, dtype: str) -> str:
    name = field.lower()
    label = _humanize_field_name(field)

    if any(token in name for token in ["date", "time", "timestamp", "day", "month", "year"]):
        return f"{label}: when something happened."
    if any(token in name for token in ["price", "revenue", "sales", "cost", "amount", "income"]):
        return f"{label}: money value related to business performance."
    if any(token in name for token in ["count", "qty", "quantity", "number", "visits", "clicks"]):
        return f"{label}: how many events or items were recorded."
    if any(token in name for token in ["rate", "ratio", "pct", "percent", "conversion"]):
        return f"{label}: a percentage or proportion."
    if any(token in name for token in ["score", "rating", "index"]):
        return f"{label}: a score that compares quality or performance."
    if any(token in name for token in ["region", "country", "city", "state", "location"]):
        return f"{label}: where the activity came from."
    if any(token in name for token in ["segment", "category", "type", "channel", "device", "source"]):
        return f"{label}: a grouping label used to compare different groups."
    if any(token in name for token in ["id", "uuid", "guid"]):
        return f"{label}: a unique identifier."
    if dtype.startswith("int") or dtype.startswith("float"):
        return f"{label}: a numeric measurement."
    return f"{label}: a descriptive field used to label records."


def _build_field_dictionary(df: pd.DataFrame) -> list[dict[str, Any]]:
    dictionary = []
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        example = None
        if not non_null.empty:
            sample = non_null.iloc[0]
            if isinstance(sample, (np.generic,)):
                sample = sample.item()
            example = str(sample)
        dictionary.append(
            {
                "field": col,
                "plain_meaning": _field_plain_meaning(col, str(series.dtype)),
                "dtype": str(series.dtype),
                "example": example,
            }
        )
    return dictionary


def _correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    valid = pd.DataFrame({"cat": categories, "val": values}).dropna()
    if len(valid) < 12:
        return None

    groups = valid.groupby("cat")["val"]
    if groups.ngroups < 2:
        return None

    global_mean = valid["val"].mean()
    ss_between = sum(len(group) * (group.mean() - global_mean) ** 2 for _, group in groups)
    ss_total = sum((valid["val"] - global_mean) ** 2)

    if ss_total == 0:
        return 0.0

    eta_squared = ss_between / ss_total
    return float(sqrt(max(0.0, eta_squared)))


def _cramers_v(cat_a: pd.Series, cat_b: pd.Series) -> float | None:
    valid = pd.DataFrame({"a": cat_a, "b": cat_b}).dropna()
    if len(valid) < 12:
        return None

    contingency = pd.crosstab(valid["a"], valid["b"])
    if contingency.empty:
        return None

    observed = contingency.to_numpy(dtype=float)
    n = observed.sum()
    if n <= 0:
        return None

    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = (row_sums @ col_sums) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)

    r, k = observed.shape
    if min(r, k) <= 1:
        return None

    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    r_corr = r - ((r - 1) ** 2) / max(n - 1, 1)
    k_corr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return None

    return float(sqrt(phi2_corr / denom))


def _clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    before_rows = len(df)
    df = df.copy()

    df.columns = [_normalize_name(col) for col in df.columns]

    missing_before = int(df.isna().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == "object":
            numeric_try = pd.to_numeric(df[col], errors="coerce")
            if numeric_try.notna().mean() > 0.8:
                df[col] = numeric_try

    for col in df.columns:
        if "date" in col or "time" in col:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.65:
                df[col] = parsed

    for col in df.select_dtypes(include=["number"]).columns:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    for col in df.select_dtypes(include=["object"]).columns:
        mode = df[col].mode(dropna=True)
        fill = mode.iloc[0] if len(mode) else "unknown"
        df[col] = df[col].fillna(fill)

    missing_after = int(df.isna().sum().sum())
    clean_meta = {
        "rows_before": before_rows,
        "rows_after": int(len(df)),
        "duplicates_removed": duplicates_before,
        "missing_before": missing_before,
        "missing_after": missing_after,
    }
    return df, clean_meta


def _is_low_value_column(series: pd.Series, row_count: int) -> bool:
    non_null_ratio = series.notna().mean()
    unique_count = series.nunique(dropna=True)

    if non_null_ratio < 0.25:
        return True

    col_name = str(series.name)
    if any(token in col_name for token in ["id", "uuid", "guid", "index"]):
        if unique_count >= max(20, int(0.9 * row_count)):
            return True

    if series.dtype == "object":
        avg_len = series.dropna().astype(str).str.len().mean() if non_null_ratio > 0 else 0
        if unique_count >= max(20, int(0.95 * row_count)) and avg_len > 12:
            return True

    if unique_count <= 1:
        return True

    return False


def _select_relevant_fields(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    removed_columns: list[str] = []
    kept_columns: list[str] = []

    for col in df.columns:
        if _is_low_value_column(df[col], len(df)):
            removed_columns.append(col)
        else:
            kept_columns.append(col)

    if not kept_columns:
        kept_columns = list(df.columns)
        removed_columns = []

    filtered = df[kept_columns].copy()

    numeric_cols = list(filtered.select_dtypes(include=["number"]).columns)
    numeric_variance: dict[str, float] = {}
    for col in numeric_cols:
        variance = filtered[col].var()
        numeric_variance[col] = float(variance) if pd.notna(variance) else 0.0

    ranked_numeric = [x[0] for x in sorted(numeric_variance.items(), key=lambda item: item[1], reverse=True)]

    categorical_cols = list(filtered.select_dtypes(include=["object"]).columns)
    ranked_categorical = sorted(
        categorical_cols,
        key=lambda col: filtered[col].nunique(dropna=True),
        reverse=True,
    )

    key_fields = (ranked_numeric[:5] + ranked_categorical[:3])[:8]

    return filtered, {
        "removed_columns": removed_columns,
        "kept_columns": kept_columns,
        "key_fields": key_fields,
    }


def _profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    date_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)

    columns = []
    for col in df.columns:
        columns.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique": int(df[col].nunique(dropna=True)),
            }
        )

    return {
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "numeric_columns": numeric_cols,
        "date_columns": date_cols,
        "columns": columns,
    }


def _discover_relationships(df: pd.DataFrame) -> dict[str, Any]:
    numeric_all = list(df.select_dtypes(include=["number"]).columns)
    numeric_all = [col for col in numeric_all if df[col].nunique(dropna=True) > 4]

    cat_all = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    cat_all = [col for col in cat_all if 2 <= df[col].nunique(dropna=True) <= 30]

    numeric_ranked = sorted(numeric_all, key=lambda c: df[c].var() if pd.notna(df[c].var()) else 0.0, reverse=True)[:12]
    cat_ranked = sorted(cat_all, key=lambda c: df[c].nunique(dropna=True), reverse=True)[:8]

    numeric_pairs: list[dict[str, Any]] = []
    for col_a, col_b in combinations(numeric_ranked, 2):
        valid = df[[col_a, col_b]].dropna()
        if len(valid) < 15:
            continue

        pearson = valid[col_a].corr(valid[col_b], method="pearson")
        # Compute Spearman as Pearson on ranks to avoid requiring scipy.
        spearman = valid[col_a].rank(method="average").corr(
            valid[col_b].rank(method="average"),
            method="pearson",
        )
        if pd.isna(pearson) or pd.isna(spearman):
            continue

        score = max(abs(float(pearson)), abs(float(spearman)))
        numeric_pairs.append(
            {
                "type": "numeric_numeric",
                "field_a": col_a,
                "field_b": col_b,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "score": float(score),
                "direction": "same direction" if pearson >= 0 else "opposite direction",
                "sample_size": int(len(valid)),
            }
        )

    numeric_pairs.sort(key=lambda item: item["score"], reverse=True)

    categorical_numeric: list[dict[str, Any]] = []
    for cat_col in cat_ranked:
        for num_col in numeric_ranked:
            eta = _correlation_ratio(df[cat_col], df[num_col])
            if eta is None:
                continue
            categorical_numeric.append(
                {
                    "type": "categorical_numeric",
                    "field_a": cat_col,
                    "field_b": num_col,
                    "eta": float(eta),
                    "score": float(eta),
                    "sample_size": int(df[[cat_col, num_col]].dropna().shape[0]),
                }
            )

    categorical_numeric.sort(key=lambda item: item["score"], reverse=True)

    categorical_pairs: list[dict[str, Any]] = []
    for col_a, col_b in combinations(cat_ranked, 2):
        v = _cramers_v(df[col_a], df[col_b])
        if v is None:
            continue
        categorical_pairs.append(
            {
                "type": "categorical_categorical",
                "field_a": col_a,
                "field_b": col_b,
                "cramers_v": float(v),
                "score": float(v),
                "sample_size": int(df[[col_a, col_b]].dropna().shape[0]),
            }
        )

    categorical_pairs.sort(key=lambda item: item["score"], reverse=True)

    all_ranked = [*numeric_pairs[:10], *categorical_numeric[:10], *categorical_pairs[:10]]
    all_ranked.sort(key=lambda item: item["score"], reverse=True)

    return {
        "numeric_pairs": numeric_pairs[:10],
        "categorical_numeric": categorical_numeric[:10],
        "categorical_pairs": categorical_pairs[:10],
        "top_relationships": all_ranked[:12],
        "numeric_fields_considered": numeric_ranked,
        "categorical_fields_considered": cat_ranked,
    }


def _distribution_diagnostics(df: pd.DataFrame, numeric_cols: list[str]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for col in numeric_cols[:10]:
        series = df[col].dropna()
        if len(series) < 12:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)
        outlier_rate = float(((series < lower) | (series > upper)).mean())

        diagnostics.append(
            {
                "field": col,
                "skew": _safe_number(series.skew()),
                "kurtosis": _safe_number(series.kurtosis()),
                "outlier_rate": _safe_number(outlier_rate),
                "recommendation": (
                    "Use robust scaling or winsorization before modeling."
                    if outlier_rate > 0.08 or abs(float(series.skew())) > 1.0
                    else "Distribution is stable for first-pass modeling."
                ),
            }
        )

    diagnostics.sort(key=lambda item: (item["outlier_rate"] or 0.0), reverse=True)
    return diagnostics


def _fit_regression(train_df: pd.DataFrame, target: str, predictors: list[str]) -> dict[str, Any] | None:
    if len(train_df) < max(24, len(predictors) * 5):
        return None

    x = train_df[predictors].to_numpy(dtype=float)
    y = train_df[target].to_numpy(dtype=float)

    x_design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(x_design, y, rcond=None)

    preds = x_design @ beta
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    n = len(y)
    p = len(predictors)
    adjusted_r2 = 1.0 - ((1.0 - r2) * (n - 1) / max(n - p - 1, 1))

    return {
        "beta": beta,
        "r2_train": float(r2),
        "adjusted_r2_train": float(adjusted_r2),
    }


def _evaluate_regression(test_df: pd.DataFrame, target: str, predictors: list[str], beta: np.ndarray) -> float | None:
    if len(test_df) < 8:
        return None

    x = test_df[predictors].to_numpy(dtype=float)
    y = test_df[target].to_numpy(dtype=float)
    x_design = np.column_stack([np.ones(len(x)), x])

    preds = x_design @ beta
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return None
    return float(1.0 - (ss_res / ss_tot))


def _dependency_modeling(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    numeric_cols = [col for col in numeric_cols if df[col].nunique(dropna=True) > 4]
    if len(numeric_cols) < 2:
        return {
            "status": "not_enough_numeric_fields",
            "message": "At least two numeric fields are needed for dependency modeling.",
            "models": [],
        }

    var_ranked = sorted(numeric_cols, key=lambda c: df[c].var() if pd.notna(df[c].var()) else 0.0, reverse=True)

    models = []
    for target in var_ranked[:6]:
        candidate_predictors = [col for col in var_ranked if col != target]
        if not candidate_predictors:
            continue

        corr_scores = []
        for pred in candidate_predictors:
            valid = df[[target, pred]].dropna()
            if len(valid) < 15:
                continue
            corr = valid[target].rank(method="average").corr(
                valid[pred].rank(method="average"),
                method="pearson",
            )
            if pd.notna(corr):
                corr_scores.append((pred, abs(float(corr))))

        corr_scores.sort(key=lambda item: item[1], reverse=True)
        predictors = [name for name, _ in corr_scores[:6]]
        if not predictors:
            continue

        model_df = df[[target, *predictors]].dropna().sample(frac=1.0, random_state=42)
        if len(model_df) < max(30, len(predictors) * 6):
            continue

        split = int(len(model_df) * 0.8)
        train_df = model_df.iloc[:split]
        test_df = model_df.iloc[split:]

        fitted = _fit_regression(train_df, target, predictors)
        if not fitted:
            continue

        test_r2 = _evaluate_regression(test_df, target, predictors, fitted["beta"])

        coeffs = []
        for idx, pred in enumerate(predictors):
            coeff = float(fitted["beta"][idx + 1])
            coeffs.append(
                {
                    "field": pred,
                    "coefficient": coeff,
                    "impact_direction": "positive" if coeff >= 0 else "negative",
                }
            )

        coeffs.sort(key=lambda item: abs(item["coefficient"]), reverse=True)

        models.append(
            {
                "target": target,
                "predictors": predictors,
                "rows": int(len(model_df)),
                "r2_train": fitted["r2_train"],
                "adjusted_r2_train": fitted["adjusted_r2_train"],
                "r2_test": test_r2,
                "coefficients": coeffs,
            }
        )

    if not models:
        return {
            "status": "insufficient_rows",
            "message": "Not enough complete rows for stable dependency modeling.",
            "models": [],
        }

    models.sort(key=lambda item: item["r2_test"] if item["r2_test"] is not None else -1.0, reverse=True)
    return {
        "status": "ok",
        "message": "Dependency modeling completed.",
        "models": models[:4],
    }


def _chart_payload(df: pd.DataFrame, relationship_bundle: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    charts: list[dict[str, Any]] = []
    visual_audit: list[dict[str, Any]] = []

    numeric_priority = relationship_bundle["numeric_fields_considered"][:4]

    for col in numeric_priority:
        sample = df[col].dropna()
        if sample.empty:
            continue
        coverage = float(len(sample) / max(len(df), 1))
        bins = np.histogram(sample, bins=min(14, max(4, int(np.sqrt(len(sample))))))
        x_edges = bins[1]
        counts = bins[0]
        points = [
            {"bin": f"{x_edges[i]:.2f} - {x_edges[i + 1]:.2f}", "count": int(counts[i])}
            for i in range(len(counts))
        ]
        charts.append(
            {
                "id": f"hist_{col}",
                "title": f"Distribution of {col}",
                "type": "bar",
                "x": "bin",
                "y": "count",
                "data": points,
                "meta": {"records_used": int(len(sample)), "coverage_ratio": coverage, "sampling": "full_non_null"},
            }
        )
        visual_audit.append(
            {
                "chart_id": f"hist_{col}",
                "status": "ok",
                "checks": [
                    "Histogram uses full non-null records.",
                    "Bar chart starts from count data per bin.",
                ],
                "warnings": [],
            }
        )

    for idx, rel in enumerate(relationship_bundle["numeric_pairs"][:3]):
        x_col, y_col = rel["field_a"], rel["field_b"]
        scatter_df = df[[x_col, y_col]].dropna()
        total_points = len(scatter_df)
        if total_points == 0:
            continue
        sample_size = min(600, total_points)
        scatter_df = scatter_df.sample(n=sample_size, random_state=42) if total_points > sample_size else scatter_df
        scatter_data = scatter_df.to_dict(orient="records")
        coverage = float(sample_size / total_points) if total_points > 0 else 0.0
        charts.append(
            {
                "id": f"scatter_{idx}",
                "title": f"{x_col} vs {y_col}",
                "type": "scatter",
                "x": x_col,
                "y": y_col,
                "data": [
                    {x_col: _safe_number(item[x_col]), y_col: _safe_number(item[y_col])}
                    for item in scatter_data
                ],
                "meta": {
                    "records_used": int(sample_size),
                    "records_available": int(total_points),
                    "coverage_ratio": coverage,
                    "sampling": "random_sample" if total_points > sample_size else "full",
                },
            }
        )
        warnings = []
        if coverage < 0.35:
            warnings.append("Scatter uses a sample of available points to keep the chart readable.")
        visual_audit.append(
            {
                "chart_id": f"scatter_{idx}",
                "status": "ok",
                "checks": [
                    "Scatter sampling is random to avoid ordering bias.",
                    "Axes map directly to raw numeric fields.",
                ],
                "warnings": warnings,
            }
        )

    date_cols = list(df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    if date_cols and numeric_priority:
        dcol, ncol = date_cols[0], numeric_priority[0]
        grouped = (
            df[[dcol, ncol]]
            .dropna()
            .assign(day=lambda x: x[dcol].dt.date.astype(str))
            .groupby("day", as_index=False)[ncol]
            .mean()
            .sort_values("day")
        )
        if not grouped.empty:
            if len(grouped) > 220:
                idxs = np.linspace(0, len(grouped) - 1, 220, dtype=int)
                grouped = grouped.iloc[idxs]
            charts.append(
                {
                    "id": "trend_time",
                    "title": f"Trend of {ncol} over time",
                    "type": "line",
                    "x": "day",
                    "y": ncol,
                    "data": [
                        {"day": row["day"], ncol: _safe_number(row[ncol])}
                        for _, row in grouped.iterrows()
                    ],
                    "meta": {
                        "records_used": int(len(grouped)),
                        "sampling": "even_time_downsample" if len(grouped) >= 220 else "full_daily_series",
                    },
                }
            )
            visual_audit.append(
                {
                    "chart_id": "trend_time",
                    "status": "ok",
                    "checks": [
                        "Time series is sorted by date before plotting.",
                        "Downsampling is evenly spaced to preserve trend shape.",
                    ],
                    "warnings": [],
                }
            )

    return charts, visual_audit


def _plain_language_story(
    profile: dict[str, Any],
    clean_meta: dict[str, Any],
    key_fields: list[str],
    removed_columns: list[str],
    relationship_bundle: dict[str, Any],
    modeling: dict[str, Any],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = profile["rows"]
    columns_count = profile["columns_count"]

    context = (
        f"We reviewed {rows} records across {columns_count} useful fields. "
        f"Focus fields: {', '.join(key_fields) if key_fields else 'most informative columns'}"
    )

    top_relationships = relationship_bundle["top_relationships"][:3]
    relationship_notes = []
    for rel in top_relationships:
        pct = int(round(rel["score"] * 100))
        if rel["type"] == "numeric_numeric":
            relationship_notes.append(
                f"{rel['field_a']} and {rel['field_b']} are linked ({pct}% strength; {rel['direction']})."
            )
        elif rel["type"] == "categorical_numeric":
            relationship_notes.append(
                f"{rel['field_a']} has a measurable effect on {rel['field_b']} ({pct}% effect size)."
            )
        else:
            relationship_notes.append(
                f"{rel['field_a']} and {rel['field_b']} are associated categories ({pct}% strength)."
            )

    if not relationship_notes:
        relationship_notes = ["No strong field dependencies were detected."]

    distribution_note = "Distribution checks look stable for first-pass modeling."
    if diagnostics and (diagnostics[0].get("outlier_rate") or 0) >= 0.08:
        distribution_note = (
            f"{diagnostics[0]['field']} has elevated outliers; use robust modeling for high-stakes decisions."
        )

    findings = [
        "EDA followed best practices: data quality cleanup, field pruning, and multi-method dependency testing.",
        *relationship_notes,
        distribution_note,
    ]

    if modeling.get("status") == "ok":
        best_model = modeling["models"][0]
        model_note = (
            f"Best dependency model predicts {best_model['target']} with test R² "
            f"{(best_model['r2_test'] or 0):.2f}."
        )
    else:
        model_note = modeling.get("message", "Modeling could not be completed reliably.")

    risks = []
    if rows < 50:
        risks.append("Small sample size can make relationships unstable.")
    if len(relationship_bundle["numeric_fields_considered"]) < 2:
        risks.append("Limited numeric diversity restricts dependency testing.")
    risks.append(model_note)

    conclusion = model_note

    return {
        "context": context,
        "data_health": [
            f"Removed {clean_meta['duplicates_removed']} duplicate rows.",
            f"Missing values reduced from {clean_meta['missing_before']} to {clean_meta['missing_after']}.",
            (
                f"Removed {len(removed_columns)} low-value fields."
                if removed_columns
                else "No low-value fields removed."
            ),
        ],
        "key_findings": findings[:6],
        "risks": risks,
        "conclusion": conclusion,
        "next_steps": [
            "Validate strongest dependencies with domain logic before causal claims.",
            "Re-run this analysis per segment (region/product/cohort) to compare relationships.",
            "Monitor model quality as new data arrives and retrain when drift appears.",
        ],
        "key_fields": key_fields,
        "removed_fields": removed_columns,
        "relationships": relationship_bundle["top_relationships"],
    }


def _build_simple_english_story(
    profile: dict[str, Any],
    clean_meta: dict[str, Any],
    key_fields: list[str],
    field_dictionary: list[dict[str, Any]],
    removed_columns: list[str],
    relationship_bundle: dict[str, Any],
    modeling: dict[str, Any],
) -> dict[str, Any]:
    top_relationships = relationship_bundle.get("top_relationships", [])[:3]
    top_model = modeling.get("models", [None])[0] if modeling.get("status") == "ok" else None

    if top_model:
        model_sentence = (
            f"The clearest pattern centers on {_humanize_field_name(top_model['target'])}. "
            f"Our model explains about {int(round((top_model.get('r2_test') or 0.0) * 100))}% "
            "of the changes in that field."
        )
    else:
        model_sentence = "We can see useful patterns, but the dataset is not strong enough for confident prediction."

    if top_relationships:
        relationship_sentences = []
        for rel in top_relationships:
            strength = int(round((rel.get("score") or 0.0) * 100))
            if rel.get("type") == "numeric_numeric":
                relationship_sentences.append(f"When {_humanize_field_name(rel['field_a'])} changes, {_humanize_field_name(rel['field_b'])} usually changes too.")
            elif rel.get("type") == "categorical_numeric":
                relationship_sentences.append(f"Different {_humanize_field_name(rel['field_a'])} groups show clear differences in {_humanize_field_name(rel['field_b'])}.")
            else:
                relationship_sentences.append(f"{_humanize_field_name(rel['field_a'])} and {_humanize_field_name(rel['field_b'])} are meaningfully connected categories.")
    else:
        relationship_sentences = ["No strong relationships were found across the main fields."]

    field_meanings = [
        f"{item['plain_meaning']}" + (f" Example value: {item['example']}." if item.get("example") else "")
        for item in field_dictionary[:8]
    ]

    simple_story = [
        f"We reviewed {profile['rows']} website records and focused on the fields that matter most.",
        "Below is a plain-English guide to what each key field means:",
        *field_meanings,
    ]

    limitations = []
    if profile["rows"] < 50:
        limitations.append("Small sample size can reduce stability of discovered patterns.")
    if len(relationship_bundle.get("numeric_fields_considered", [])) < 2:
        limitations.append("Limited numeric fields reduce depth of dependency modeling.")
    if modeling.get("status") != "ok":
        limitations.append(modeling.get("message", "Dependency modeling could not complete reliably."))
    if not limitations:
        limitations.append("No major statistical blockers were detected, but results should still be validated.")
    return {
        "field_guide": " ".join(simple_story[:3] + field_meanings[:3]),
        "what_we_learned": " ".join(
            [
                f"Data cleanup removed {clean_meta['duplicates_removed']} duplicate rows and reduced missing values from {clean_meta['missing_before']} to {clean_meta['missing_after']}.",
                *relationship_sentences,
                (
                    f"We removed {len(removed_columns)} low-value fields so the results stay focused."
                    if removed_columns
                    else "No low-value fields needed to be removed."
                ),
            ]
        ),
        "final_takeaway": model_sentence,
        "important_caveats": " ".join(limitations),
    }


def _rewrite_story_with_gemini(simple_story: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not api_key:
        return simple_story

    prompt_payload = {
        "system_instruction": """
You are an expert Data Journalist. Your goal is to translate raw statistical analysis into a clear, engaging plain English story for a non-technical business audience.

You must return a valid JSON object.

STRICT WRITING RULES:
1. Narrative Flow: Every value in the JSON must be a coherent paragraph. Do not use bullet points, lists, or sentence fragments.
2. Plain English: Strictly avoid statistical jargon (e.g., do not say 'p-value', 'multicollinearity', or 'heteroscedasticity'). Instead, describe what the data is doing (e.g., say 'strong relationship', 'predictable pattern', or 'highly consistent').
3. Tone: Helpful, objective, and storytelling.
4. Analogies: Use simple real-world analogies to explain complex math concepts in the 'field_guide'.
""",
        "prompt_template": """
Analyze the following statistical data and return a JSON object with exactly these four keys.

Input Data: {input_data}

Required JSON Structure:
{
    "field_guide": "A narrative paragraph setting the scene. Explain the goal of the analysis and the methodology using a simple analogy. (e.g., 'Think of this model like a weather forecast...')",

    "what_we_learned": "A narrative paragraph explaining the results. Tell the story of the relationships in the data. Explain what influences what, and by how much, in plain language.",

    "final_takeaway": "A short, punchy paragraph (2-3 sentences) summarizing the single most important actionable business insight.",

    "important_caveats": "A continuous prose paragraph explaining the limitations. Do not list them. Explain *why* the reader should be cautious (e.g., sample size issues, correlation vs causation) in a flow."
}
""",
        "generationConfig": {
            "temperature": 0.4,
            "response_mime_type": "application/json",
        },
    }

    input_data = json.dumps(simple_story)
    prompt_text = prompt_payload["prompt_template"].replace("{input_data}", input_data)
    body = {
        "system_instruction": {
            "parts": [{"text": prompt_payload["system_instruction"].strip()}],
        },
        "contents": [
            {
                "parts": [{"text": prompt_text.strip()}],
            }
        ],
        "generationConfig": prompt_payload["generationConfig"],
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{urllib_parse.quote(model)}:generateContent"
    )
    req = urllib_request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib_error.URLError, TimeoutError, json.JSONDecodeError):
        return simple_story

    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        rewritten = json.loads(cleaned)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return simple_story

    return {
        "field_guide": rewritten.get(
            "field_guide",
            "This analysis reviews the most important fields and checks how they move together over time."
        ),
        "what_we_learned": rewritten.get(
            "what_we_learned",
            "The data shows several meaningful relationships, but strength varies by field and by data coverage."
        ),
        "final_takeaway": rewritten.get(
            "final_takeaway",
            simple_story.get("final_takeaway", "The strongest pattern is useful for direction, but validate before acting.")
        ),
        "important_caveats": rewritten.get(
            "important_caveats",
            "These results describe patterns in this dataset and should be validated with domain context before decisions are finalized."
        ),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/download-cleaned/{analysis_id}")
def download_cleaned(analysis_id: str) -> Response:
    cached = ANALYSIS_CACHE.get(analysis_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Analysis id not found.")

    return Response(
        content=cached["csv"],
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={cached['file_name']}"},
    )


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if not file.filename.lower().endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Please upload a CSV file."})

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=413,
            content={"error": "File is too large. Maximum allowed size is 5 MB."},
        )
    try:
        text = content.decode("utf-8", errors="replace")
        df = pd.read_csv(StringIO(text))
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": f"CSV parsing failed: {exc}"})

    if df.empty:
        return JSONResponse(status_code=400, content={"error": "CSV is empty."})

    cleaned_df, clean_meta = _clean_dataframe(df)
    focused_df, field_meta = _select_relevant_fields(cleaned_df)
    profile = _profile(focused_df)

    relationship_bundle = _discover_relationships(focused_df)
    diagnostics = _distribution_diagnostics(focused_df, relationship_bundle["numeric_fields_considered"])
    modeling = _dependency_modeling(focused_df, relationship_bundle["numeric_fields_considered"])

    field_dictionary = _build_field_dictionary(focused_df)
    charts, visual_audit = _chart_payload(focused_df, relationship_bundle)
    storyboard = _plain_language_story(
        profile,
        clean_meta,
        field_meta["key_fields"],
        field_meta["removed_columns"],
        relationship_bundle,
        modeling,
        diagnostics,
    )
    simple_story = _build_simple_english_story(
        profile,
        clean_meta,
        field_meta["key_fields"],
        field_dictionary,
        field_meta["removed_columns"],
        relationship_bundle,
        modeling,
    )
    simple_story = _rewrite_story_with_gemini(simple_story)

    preview = focused_df.head(20).copy()
    for col in preview.columns:
        if np.issubdtype(preview[col].dtype, np.datetime64):
            preview[col] = preview[col].astype(str)

    analysis_id = str(uuid4())
    output_name = f"cleaned_{_normalize_name(file.filename.rsplit('.', 1)[0])}.csv"
    ANALYSIS_CACHE[analysis_id] = {
        "csv": focused_df.to_csv(index=False),
        "file_name": output_name,
    }

    response = {
        "analysis_id": analysis_id,
        "file_name": file.filename,
        "cleaning": {**clean_meta, "removed_columns": len(field_meta["removed_columns"])},
        "profile": profile,
        "field_analysis": {
            "key_fields": field_meta["key_fields"],
            "removed_fields": field_meta["removed_columns"],
            "kept_fields": field_meta["kept_columns"],
            "field_dictionary": field_dictionary,
            "top_relationships": relationship_bundle["top_relationships"],
            "numeric_pairs": relationship_bundle["numeric_pairs"],
            "categorical_numeric": relationship_bundle["categorical_numeric"],
            "categorical_pairs": relationship_bundle["categorical_pairs"],
            "distribution_diagnostics": diagnostics,
        },
        "modeling": modeling,
        "eda_best_practices": {
            "quality_checks": storyboard["data_health"],
            "distribution_checks": diagnostics[:6],
            "relationship_checks": relationship_bundle["top_relationships"][:8],
            "modeling_check": modeling,
            "visual_checks": visual_audit,
        },
        "charts": charts,
        "storyboard": storyboard,
        "simple_story": simple_story,
        "preview": {"columns": list(preview.columns), "rows": preview.to_dict(orient="records")},
        "downloads": {"cleaned_csv": f"/download-cleaned/{analysis_id}"},
    }
    return JSONResponse(content=response)
