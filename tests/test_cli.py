from poliseye.cli import main


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "weekday x slot profile" in out
    assert "transport hub" in out


def test_benchmark_on_file(tmp_path, wide, capsys):
    long = wide.stack().rename("nr_people").reset_index()
    long["date"] = long["timestamp"].dt.strftime("%Y-%m-%d")
    long["hours"] = long["timestamp"].dt.hour.astype(float)
    long["minutes"] = long["timestamp"].dt.minute.astype(float)
    long = long.rename(columns={"cell": "cell_num"})
    f = tmp_path / "data.csv"
    long[["date", "cell_num", "nr_people", "hours", "minutes"]].to_csv(f)
    assert main(["benchmark", str(f), "--origins", "2", "--cells", "486-1258"]) == 0
    assert "persistence" in capsys.readouterr().out
