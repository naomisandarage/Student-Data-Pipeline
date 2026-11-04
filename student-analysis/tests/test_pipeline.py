import pandas as pd
import pytest

from src import pipeline


def make_df(**overrides):
    """One valid student row, with optional field overrides."""
    row = {"School": "GP", "ID": 1, "Age": 17, "StudyTime": 2, "Failures": 0,
           "Health": 3, "Absences": 4, "FallGrade": 10, "WinterGrade": 12}
    row.update(overrides)
    return pd.DataFrame([row])


def test_clean_keeps_valid_rows():
    df, report = pipeline.clean(make_df())
    assert len(df) == 1
    assert report["rows_out"] == 1


def test_clean_removes_duplicate_ids():
    df, report = pipeline.clean(pd.concat([make_df(), make_df()]))
    assert len(df) == 1
    assert report["dropped_duplicate_id"] == 1


def test_clean_removes_out_of_range_grades():
    df, report = pipeline.clean(pd.concat([make_df(ID=1), make_df(ID=2, WinterGrade=25)]))
    assert list(df["ID"]) == [1]
    assert report["dropped_grade_out_of_range"] == 1


def test_clean_removes_fractional_studytime():
    df, report = pipeline.clean(make_df(StudyTime=2.5))
    assert df.empty
    assert report["dropped_invalid_studytime"] == 1


def test_clean_removes_missing_values():
    df, report = pipeline.clean(make_df(Age=None))
    assert df.empty
    assert report["dropped_missing"] == 1


def test_add_features():
    df = pipeline.add_features(make_df(FallGrade=10, WinterGrade=14))
    assert df.loc[0, "GradeChange"] == 4
    assert df.loc[0, "AverageGrade"] == 12


def test_summarize_by_computes_group_means():
    df = pd.concat([make_df(ID=1, StudyTime=1, WinterGrade=10, FallGrade=10),
                    make_df(ID=2, StudyTime=1, WinterGrade=14, FallGrade=10),
                    make_df(ID=3, StudyTime=2, WinterGrade=8, FallGrade=8)])
    out = pipeline.summarize_by(pipeline.add_features(df), "StudyTime")
    group1 = out[out["StudyTime"] == 1].iloc[0]
    assert group1["students"] == 2
    assert group1["avg_winter"] == 12


def test_extract_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("School, ID\nGP, 1\n")
    with pytest.raises(ValueError):
        pipeline.extract(bad)


def test_extract_handles_spaces_after_commas(tmp_path):
    f = tmp_path / "ok.csv"
    f.write_text(", ".join(pipeline.EXPECTED_COLUMNS) + "\n"
                 "GP, 1, 18, 2, 0, 3, 6, 5, 6\n")
    df = pipeline.extract(f)
    assert df.loc[0, "School"] == "GP"
    assert df.loc[0, "Age"] == 18
