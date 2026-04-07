import sys
import argparse
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

# Using MapCode as the primary filter for consistency with your previous script
ALLOWED_MAP_CODES = {
    "AT", "BA", "BE", "BG", "CH", "CZ", "DE",
    "DK", "EE", "ES", "FI", "FR", "GB",
    "GR", "HR", "HU", "IE", "IT", "LT", "LV", "MD",
    "ME", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK"
}

DTYPE = {
    "MapCode": "category",
    "Currency": "category",
}

def process_file(path: str, chunksize: int) -> pd.DataFrame:
    # Detect separator based on expected price columns
    sample = pd.read_csv(path, nrows=1, sep=None, engine="python")
    sep = "\t" if "DateTime(UTC)" in sample.columns else ","

    # Columns specific to the Price CSV structure
    usecols = ["DateTime(UTC)", "MapCode", "Price[Currency/MWh]"]

    chunks = []
    for chunk in pd.read_csv(
        path,
        sep=sep,
        usecols=usecols,
        dtype=DTYPE,
        parse_dates=["DateTime(UTC)"],
        chunksize=chunksize,
        low_memory=False,
    ):
        # Filter by MapCode
        mask = chunk["MapCode"].isin(ALLOWED_MAP_CODES)
        chunk = chunk[mask].copy()
        
        if chunk.empty:
            continue

        # Normalize to Date for daily aggregation
        chunk["Date"] = chunk["DateTime(UTC)"].dt.normalize()

        # Aggregate: Mean price per day per MapCode
        agg = (
            chunk.groupby(["Date", "MapCode"], observed=True)["Price[Currency/MWh]"]
            .mean()
            .reset_index()
        )
        chunks.append(agg)

    if not chunks:
        print(f"  [!] No useful data found in {path}")
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    # Final aggregation to handle duplicates across multiple files
    df = (
        df.groupby(["Date", "MapCode"], observed=True)["Price[Currency/MWh]"]
        .mean()
        .reset_index()
    )
    df["Column"] = df["MapCode"].astype(str) + "_Price"

    pivot = df.pivot_table(
        index="Date",
        columns="Column",
        values="Price[Currency/MWh]",
        aggfunc="mean",
        fill_value=0,
    )
    pivot.index.name = "Date"
    pivot.columns.name = None
    return pivot.sort_index(axis=1)


def save_excel(pivot: pd.DataFrame, output_path: str) -> None:
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pivot.to_excel(writer, sheet_name="Day-Ahead Prices")
        ws = writer.sheets["Day-Ahead Prices"]

        # Styling (Match your previous blue theme)
        header_fill = PatternFill("solid", start_color="2D6A9F", end_color="2D6A9F")
        header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        cell_font = Font(name="Arial", size=10)

        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        ws.column_dimensions["A"].width = 14
        for col_idx in range(2, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = 16

        for row in ws.iter_rows(min_row=2):
            row[0].font = cell_font
            for cell in row[1:]:
                cell.font = cell_font
                cell.number_format = "#,##0.00"  # Prices usually need 2 decimals

        ws.freeze_panes = "B2"

    print(f"\n✅ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Pivot CSV Day-Ahead Prices ENTSO-E")
    parser.add_argument("files", nargs="+", help="One or more CSV files")
    parser.add_argument("-o", "--output", default="prices_pivot.xlsx")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    all_agg = []
    for path in args.files:
        print(f"→ Reading: {path}")
        agg = process_file(path, args.chunksize)
        if not agg.empty:
            all_agg.append(agg)

    if not all_agg:
        print("No data found.")
        sys.exit(1)

    combined = pd.concat(all_agg, ignore_index=True)
    pivot = build_pivot(combined)

    print(f"\nPivot shape: {pivot.shape} ({pivot.index.min().date()} → {pivot.index.max().date()})")
    print(pivot.head(3).to_string())

    save_excel(pivot, args.output)


if __name__ == "__main__":
    main()