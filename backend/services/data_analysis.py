from pathlib import Path
import pandas as pd
import plotly.express as px
from backend.config import UPLOAD_DIR

def list_csv_files() -> list[str]:
    return sorted(p.name for p in UPLOAD_DIR.glob("*.csv") if p.is_file())

def _safe_csv(filename: str) -> Path:
    name = Path(filename).name
    if name != filename or not name.lower().endswith(".csv"):
        raise ValueError("Choose an uploaded CSV filename.")
    path = (UPLOAD_DIR / name).resolve()
    if path.parent != UPLOAD_DIR.resolve() or not path.is_file():
        raise FileNotFoundError("CSV file not found. Upload it on the Data Analysis page first.")
    return path

def analyze_csv(filename: str, column: str | None = None) -> dict:
    path = _safe_csv(filename)
    df = pd.read_csv(path)
    result = {
        "filename": path.name,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
        "numeric_summary": df.describe(include="number").replace({float("nan"): None}).to_dict(),
    }
    if column:
        if column not in df.columns:
            result["column_error"] = f"Column {column!r} not found."
        else:
            counts = df[column].astype(str).value_counts().head(20)
            result["value_counts"] = {str(k): int(v) for k, v in counts.items()}
    return result

def build_chart(filename: str, x_column: str, y_column: str | None, chart_type: str):
    path = _safe_csv(filename)
    df = pd.read_csv(path)
    if x_column not in df.columns:
        raise ValueError(f"Unknown X column: {x_column}")
    if y_column and y_column not in df.columns:
        raise ValueError(f"Unknown Y column: {y_column}")
    if chart_type == "bar":
        fig = px.bar(df, x=x_column, y=y_column, title=f"{y_column or x_column} by {x_column}")
    elif chart_type == "scatter" and y_column:
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{y_column} vs {x_column}")
    elif chart_type == "line" and y_column:
        fig = px.line(df, x=x_column, y=y_column, title=f"{y_column} over {x_column}")
    else:
        fig = px.histogram(df, x=x_column, title=f"Distribution of {x_column}")
    return fig
