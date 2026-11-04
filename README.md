# Student Data Pipeline

A reproducible Extract -> Transform -> Analyze -> Load pipeline built with pandas.
It is a rebuild of a first-year team project (ECOR 1042) that loaded, sorted and
plotted student data using hand-written Python.

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python -m src.pipeline
python -m src.pipeline --input data/student-mat.csv --output output
```

## Test
```
pytest
```

## What it does
1. **Extract** - reads `data/student-mat.csv` and checks the expected columns exist.
2. **Transform** - removes missing values, duplicate IDs, out-of-range grades (outside 0-20)
   and invalid study-time values; adds `GradeChange` and `AverageGrade`.
   Every removal is counted in `output/cleaning_report.csv`.
3. **Analyze** - average grades by school, study time and failures; correlation matrix.
4. **Load** - writes cleaned data, summary tables and three charts to `output/`.

## Design notes
- Each stage is a small function so it can be tested on its own.
- Input and output paths are command-line arguments (nothing hardcoded).
- Outliers (e.g. very high absences) are reported but kept, since unusual is not the same as wrong.
