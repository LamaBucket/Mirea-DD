from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Мини-CLI для EDA CSV-файлов")


def _load_csv(
    path: Path,
    sep: str = ",",
    encoding: str = "utf-8",
) -> pd.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не удалось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Напечатать краткий обзор датасета:
    - размеры;
    - типы;
    - простая табличка по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV-файлу."),
    out_dir: str = typer.Option("reports", help="Каталог для отчёта."),
    sep: str = typer.Option(",", help="Разделитель в CSV."),
    encoding: str = typer.Option("utf-8", help="Кодировка файла."),
    max_hist_columns: int = typer.Option(
        6, help="Максимум числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5, help="Количество top-значений для категориальных признаков."
    ),
    title: str = typer.Option(
        "EDA-отчёт", help="Заголовок отчёта."
    ),
    min_missing_share: float = typer.Option(
        0.3,
        help="Порог доли пропусков, выше которого колонка считается проблемной.",
    ),
) -> None:
    """
    Сгенерировать полный EDA-отчёт с настраиваемыми параметрами.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1. Базовые расчёты
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2. Качество данных
    quality_flags = compute_quality_flags(summary, missing_df, df)

    # 3. Табличные артефакты
    summary_df.to_csv(out_root / "summary.csv", index=False)
    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)
    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)
    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4. Проблемные колонки по пропускам
    problematic_missing = (
        missing_df[missing_df["missing_share"] >= min_missing_share]
        if not missing_df.empty
        else pd.DataFrame()
    )

    # 5. Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Исходный файл: `{Path(path).name}`\n\n")
        f.write(
            f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n"
        )

        f.write("## Параметры генерации отчёта\n\n")
        f.write(f"- max_hist_columns = `{max_hist_columns}`\n")
        f.write(f"- top_k_categories = `{top_k_categories}`\n")
        f.write(f"- min_missing_share = `{min_missing_share:.0%}`\n\n")

        f.write("## Качество данных (эвристики)\n\n")
        f.write(f"- Оценка качества: **{quality_flags['quality_score']:.2f}**\n")
        f.write(
            f"- Макс. доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n"
        )
        f.write(
            f"- Константные колонки: **{quality_flags['has_constant_columns']}**\n"
        )
        f.write(
            f"- Высокая кардинальность категорий: "
            f"**{quality_flags['has_high_cardinality_categoricals']}**\n"
        )
        f.write(
            f"- Подозрительные ID-дубликаты: "
            f"**{quality_flags['has_suspicious_id_duplicates']}**\n\n"
        )

        f.write("## Колонки с большим числом пропусков\n\n")
        if problematic_missing.empty:
            f.write("Такие колонки не обнаружены.\n\n")
        else:
            f.write(
                f"Колонки с долей пропусков ≥ {min_missing_share:.0%}:\n\n"
            )
            for name, row in problematic_missing.iterrows():
                f.write(
                    f"- **{name}**: {row['missing_share']:.2%}\n"
                )
            f.write("\n")

        f.write("## Дополнительные материалы\n\n")
        f.write("- `summary.csv` — сводка по колонкам\n")
        f.write("- `missing.csv` — таблица пропусков\n")
        f.write("- `correlation.csv` — корреляция числовых признаков\n")
        f.write("- `top_categories/` — top-значения категорий\n\n")

    # 6. Графики
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сгенерирован в каталоге: {out_root}")
    typer.echo(f"- Markdown: {md_path}")


if __name__ == "__main__":
    app()