from typer.testing import CliRunner

from mcbuild.cli import app

runner = CliRunner()


def test_cli_fake_llm_offline_run(tmp_path):
    out_dir = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "a tiny stone hut",
            "--fake-llm",
            "--display",
            "off",
            "--max-iters",
            "3",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Exported:" in result.output

    run_dirs = list(out_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert (run_dir / "prompt.txt").exists()
    assert (run_dir / "session.json").exists()
    assert (run_dir / "final.schem").exists()
    assert (run_dir / "final_blueprint.py").exists()
    assert (run_dir / "iter_01" / "blueprint.py").exists()
    assert (run_dir / "iter_02" / "render.png").exists()
    assert (run_dir / "iter_02" / "stats.json").exists()
