# Reruns every offline analysis from the cached data in out_phase1/data (no internet, no GPU).
# Usage (from the ada_contam folder):  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"
$py = ".venv\Scripts\python"
$steps = @(
    "part2_analysis.py --data out_phase1/data --out out_phase1",
    "phase1_audit.py --data out_phase1/data --out out_phase1",
    "phase2_decouple.py --data out_phase1/data --out out_phase1",
    "phase3_merges.py --data out_phase1/data --out out_phase1",
    "phase4_analyze.py --models out_phase1/phase4/models.csv --preds out_phase1/phase4/preds --out out_phase1/phase4",
    "phase5_robust.py --data out_phase1/data --out out_phase1",
    "make_manifest.py --out out_phase1"
)
foreach ($s in $steps) {
    Write-Host "=== $s"
    & $py @($s -split ' ')
    if ($LASTEXITCODE -ne 0) { throw "failed: $s" }
}
Write-Host "All analyses reran. Compare out_phase1/MANIFEST.json with the committed copy."
