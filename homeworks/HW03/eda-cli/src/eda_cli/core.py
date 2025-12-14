from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def summarize_dataset(
    df: pd.DataFrame,
    example_values_per_column: int = 3,
) -> DatasetSummary:
    """
    Полный обзор датасета по колонкам:
    - количество строк/столбцов;
    - типы;
    - пропуски;
    - количество уникальных;
    - несколько примерных значений;
    - базовые числовые статистики (для numeric).
    """
    n_rows, n_cols = df.shape
    columns: List[ColumnSummary] = []

    for name in df.columns:
        s = df[name]
        dtype_str = str(s.dtype)

        non_null = int(s.notna().sum())
        missing = n_rows - non_null
        missing_share = float(missing / n_rows) if n_rows > 0 else 0.0
        unique = int(s.nunique(dropna=True))

        # Примерные значения выводим как строки
        examples = (
            s.dropna().astype(str).unique()[:example_values_per_column].tolist()
            if non_null > 0
            else []
        )

        is_numeric = bool(ptypes.is_numeric_dtype(s))
        min_val: Optional[float] = None
        max_val: Optional[float] = None
        mean_val: Optional[float] = None
        std_val: Optional[float] = None

        if is_numeric and non_null > 0:
            min_val = float(s.min())
            max_val = float(s.max())
            mean_val = float(s.mean())
            std_val = float(s.std())

        columns.append(
            ColumnSummary(
                name=name,
                dtype=dtype_str,
                non_null=non_null,
                missing=missing,
                missing_share=missing_share,
                unique=unique,
                example_values=examples,
                is_numeric=is_numeric,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=columns)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: count/share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    total = df.isna().sum()
    share = total / len(df)
    result = (
        pd.DataFrame(
            {
                "missing_count": total,
                "missing_share": share,
            }
        )
        .sort_values("missing_share", ascending=False)
    )
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Корреляция Пирсона для числовых колонок.
    """
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных/строковых колонок считает top-k значений.
    Возвращает словарь: колонка -> DataFrame со столбцами value/count/share.
    """
    result: Dict[str, pd.DataFrame] = {}
    candidate_cols: List[str] = []

    for name in df.columns:
        s = df[name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            candidate_cols.append(name)

    for name in candidate_cols[:max_columns]:
        s = df[name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue
        share = vc / vc.sum()
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": share.values,
            }
        )
        result[name] = table

    return result


def compute_quality_flags(
    summary: DatasetSummary,
    missing_df: pd.DataFrame,
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Расширенные эвристики «качества» данных.
    """
    flags: Dict[str, Any] = {}

    # БАЗОВЫЕ ФЛАГИ (были)
    flags["too_few_rows"] = summary.n_rows < 100
    flags["too_many_columns"] = summary.n_cols > 100

    max_missing_share = (
        float(missing_df["missing_share"].max())
        if not missing_df.empty
        else 0.0
    )
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5


    # НОВЫЕ ЭВРИСТИКИ

    # 1. Константные колонки
    constant_columns = [
        c.name for c in summary.columns if c.unique <= 1
    ]
    flags["has_constant_columns"] = len(constant_columns) > 0
    flags["constant_columns"] = constant_columns

    # 2. Категориальные с высокой кардинальностью
    high_cardinality_cols = []
    for c in summary.columns:
        if not c.is_numeric and c.non_null > 0:
            if c.unique > max(100, 0.5 * summary.n_rows):
                high_cardinality_cols.append(c.name)

    flags["has_high_cardinality_categoricals"] = len(high_cardinality_cols) > 0
    flags["high_cardinality_columns"] = high_cardinality_cols

    # Далее флаги требуют доступ к DataFrame
    if df is not None and not df.empty:

        # 3. Подозрительные ID с дубликатами
        id_like_cols = [
            col for col in df.columns
            if col.lower() == "id" or col.lower().endswith("_id")
        ]

        duplicated_id_cols = []
        for col in id_like_cols:
            if df[col].notna().any():
                if df[col].duplicated().any():
                    duplicated_id_cols.append(col)

        flags["has_suspicious_id_duplicates"] = len(duplicated_id_cols) > 0
        flags["duplicated_id_columns"] = duplicated_id_cols

        # 4. Слишком много нулей в числовых колонках
        zero_heavy_cols = []
        for col in df.select_dtypes(include="number").columns:
            non_null = df[col].notna().sum()
            if non_null == 0:
                continue
            zero_share = (df[col] == 0).sum() / non_null
            if zero_share > 0.5:
                zero_heavy_cols.append(col)

        flags["has_many_zero_values"] = len(zero_heavy_cols) > 0
        flags["zero_heavy_columns"] = zero_heavy_cols

    else:
        flags["has_suspicious_id_duplicates"] = False
        flags["has_many_zero_values"] = False
        flags["duplicated_id_columns"] = []
        flags["zero_heavy_columns"] = []

    # ИНТЕГРАЛЬНЫЙ SCORE
    score = 1.0
    score -= max_missing_share

    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.1
    if flags["has_high_cardinality_categoricals"]:
        score -= 0.1
    if flags["has_many_zero_values"]:
        score -= 0.1
    if flags["has_suspicious_id_duplicates"]:
        score -= 0.15

    flags["quality_score"] = max(0.0, min(1.0, score))

    return flags

def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Превращает DatasetSummary в табличку для более удобного вывода.
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )
    return pd.DataFrame(rows)