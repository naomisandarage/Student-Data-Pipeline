"""Student data pipeline: Extract -> Transform (clean + features) -> Analyze -> Load.

Run from the project root:
    python -m src.pipeline
    python -m src.pipeline --input data/student-mat.csv --output output
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "School", "ID", "Age", "StudyTime", "Failures",
    "Health", "Absences", "FallGrade", "WinterGrade",
]
GRADE_MIN, GRADE_MAX = 0, 20
VALID_STUDY_TIMES = {1, 2, 3, 4}


# ----------------------------------------------------------------- EXTRACT
def extract(path):
    """Read the CSV into a DataFrame and confirm it has the columns we expect."""
    df = pd.read_csv(path, skipinitialspace=True)  # file has spaces after commas
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {sorted(missing)}")
    logger.info("Extracted %d rows from %s", len(df), path)
    return df


# --------------------------------------------------------------- TRANSFORM
def clean(df):
    """Drop invalid rows. Returns (clean_df, report) so every drop is accounted for."""
    report = {"rows_in": len(df)}

    df = df.dropna(subset=EXPECTED_COLUMNS)
    report["dropped_missing"] = report["rows_in"] - len(df)

    before = len(df)
    df = df.drop_duplicates(subset="ID", keep="first")
    report["dropped_duplicate_id"] = before - len(df)

    before = len(df)
    grades_ok = df[["FallGrade", "WinterGrade"]].apply(
        lambda col: col.between(GRADE_MIN, GRADE_MAX)
    ).all(axis=1)
    df = df[grades_ok]
    report["dropped_grade_out_of_range"] = before - len(df)

    before = len(df)
    df = df[df["StudyTime"].isin(VALID_STUDY_TIMES)]
    report["dropped_invalid_studytime"] = before - len(df)

    before = len(df)
    df = df[(df["Absences"] >= 0) & (df["Failures"] >= 0)]
    report["dropped_negative_counts"] = before - len(df)

    # Outliers are reported but kept: they are unusual, not necessarily wrong.
    q1, q3 = df["Absences"].quantile([0.25, 0.75])
    report["absence_outliers_kept"] = int((df["Absences"] > q3 + 1.5 * (q3 - q1)).sum())

    df = df.copy()
    df["StudyTime"] = df["StudyTime"].astype(int)
    df["School"] = df["School"].str.strip()
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def add_features(df):
    """Create derived columns used in the analysis."""
    df = df.copy()
    df["GradeChange"] = df["WinterGrade"] - df["FallGrade"]
    df["AverageGrade"] = (df["FallGrade"] + df["WinterGrade"]) / 2
    return df


# ----------------------------------------------------------------- ANALYZE
def summarize_by(df, column):
    """Average grades and group size for each value of `column`."""
    return (
        df.groupby(column)
        .agg(
            students=("ID", "count"),
            avg_fall=("FallGrade", "mean"),
            avg_winter=("WinterGrade", "mean"),
            avg_change=("GradeChange", "mean"),
        )
        .round(2)
        .reset_index()
    )


def correlations(df):
    """Correlation matrix of the numeric columns."""
    cols = ["Age", "StudyTime", "Failures", "Health", "Absences", "FallGrade", "WinterGrade"]
    return df[cols].corr().round(2)


def analyze(df):
    """Run every analysis and return the results as a dict of DataFrames."""
    return {
        "by_school": summarize_by(df, "School"),
        "by_studytime": summarize_by(df, "StudyTime"),
        "by_failures": summarize_by(df, "Failures"),
        "correlations": correlations(df),
    }


# -------------------------------------------------------------------- LOAD
def save_outputs(df, results, report, out_dir):
    """Write cleaned data, summary tables, the cleaning report and charts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "clean_students.csv", index=False)
    for name, table in results.items():
        table.to_csv(out_dir / f"{name}.csv", index=(name == "correlations"))
    pd.Series(report, name="count").to_csv(out_dir / "cleaning_report.csv", header=True)

    # Chart 1: distribution of winter grades
    fig, ax = plt.subplots()
    ax.hist(df["WinterGrade"], bins=range(0, 22, 2), edgecolor="black")
    ax.set(title="Winter grade distribution", xlabel="Grade (0-20)", ylabel="Students")
    fig.savefig(out_dir / "winter_grade_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: average winter grade by study time
    fig, ax = plt.subplots()
    by_study = results["by_studytime"]
    ax.bar(range(len(by_study)), by_study["avg_winter"])
    ax.set_xticks(range(len(by_study)), labels=by_study["StudyTime"].astype(str))
    ax.set(title="Average winter grade by study time",
           xlabel="Study time category", ylabel="Average winter grade")
    fig.savefig(out_dir / "grade_by_studytime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 3: fall vs winter grade
    fig, ax = plt.subplots()
    ax.scatter(df["FallGrade"], df["WinterGrade"], alpha=0.5)
    ax.set(title="Fall vs winter grade", xlabel="Fall grade", ylabel="Winter grade")
    fig.savefig(out_dir / "fall_vs_winter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved outputs to %s", out_dir)


# -------------------------------------------------------------------- MAIN
def run(input_path, out_dir):
    df = extract(input_path)
    df, report = clean(df)
    df = add_features(df)
    results = analyze(df)
    save_outputs(df, results, report, out_dir)
    return df, results, report


def main():
    parser = argparse.ArgumentParser(description="Student data pipeline")
    parser.add_argument("--input", default="data/student-mat.csv")
    parser.add_argument("--output", default="output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _, results, report = run(args.input, args.output)
    print("\nCleaning report:", report)
    print("\nAverage grades by study time:\n", results["by_studytime"].to_string(index=False))


if __name__ == "__main__":
    main()
