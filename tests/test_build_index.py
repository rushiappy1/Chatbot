import pandas as pd

from build_index import chunk_text, load_data_from_csv


def test_chunk_text_short_string():
    text = "hello world"
    chunks = chunk_text(text, max_tokens=50)
    assert chunks == ["hello world"]


def test_chunk_text_long_string_splits():
    text = "a" * 120
    chunks = chunk_text(text, max_tokens=50)
    # 120 chars with max 50 per chunk -> 3 chunks (50, 50, 20)
    assert len(chunks) == 3
    assert all(isinstance(c, str) for c in chunks)


def test_load_data_from_csv_builds_text(tmp_path):
    # Create a tiny CSV matching example.csv structure
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text(
        "Date,emp_id,EmployeeName,vehicleNumber,Target,mixed_waste,segregate_waste,Not_collected,Not_specified,Not_Scan,TotalHouseCount,duty_on_time,duty_off_time,working_time,DutyDurationInHours,FirstHouseScan,LastHouseScan,DumpTrip\n"  # noqa: E501
        "01-01-2025,9999,Demo User,MH00-XX-0000,500,10,50,0,0,0,60,06:00 AM,08:00 AM,120,02:00,6:05AM,7:55AM,1\n"
    )

    df = load_data_from_csv(str(csv_path))
    # One row -> at least one chunk
    assert len(df) >= 1
    assert set(df.columns) == {"id", "text"}
    # Text should contain some of the fields we expect
    t = df.iloc[0]["text"]
    assert "Demo User" in t
    assert "MH00-XX-0000" in t
