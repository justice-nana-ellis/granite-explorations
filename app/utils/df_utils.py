import pandas as pd


def df_summary(df: pd.DataFrame, filename: str) -> str:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c not in date_cols]

    lines = [
        f"File: {filename}",
        f"Rows: {len(df):,} | Columns: {len(df.columns)}",
        f"Columns: {', '.join(df.columns.tolist())}",
    ]
    if date_cols:
        for dc in date_cols:
            uniq = df[dc].dropna().unique()
            lines.append(
                f"Unique {dc} values ({len(uniq)}): {', '.join(sorted(str(v) for v in uniq))}"
            )
    if numeric_cols:
        lines.append("\nNumeric column totals (all rows):")
        for col in numeric_cols:
            col_data = df[col].dropna()
            lines.append(
                f"  {col}: total={col_data.sum():,.4f}, mean={col_data.mean():,.4f}, "
                f"min={col_data.min():,.4f}, max={col_data.max():,.4f}, count={len(col_data):,}"
            )
    if date_cols and numeric_cols:
        primary_date = date_cols[0]
        lines.append(f"\nNumeric totals grouped by {primary_date}:")
        grouped = df.groupby(primary_date)[numeric_cols].sum()
        for date_val, row in grouped.iterrows():
            row_parts = ", ".join(f"{col}={row[col]:,.2f}" for col in numeric_cols)
            lines.append(f"  {date_val}: {row_parts}")
    if cat_cols:
        lines.append("\nCategorical column summaries:")
        for col in cat_cols[:10]:
            uniq = df[col].dropna().unique()
            sample = ", ".join(str(v) for v in uniq[:10])
            lines.append(f"  {col}: {len(uniq)} unique values — e.g. {sample}")
    lines.append(f"\nFirst 5 rows:\n{df.head(5).to_string(index=False)}")
    return "\n".join(lines)
