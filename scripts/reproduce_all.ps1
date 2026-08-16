$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
New-Item -ItemType Directory -Force -Path results/raw,results/logs,results/metrics | Out-Null

function Run-Step($n, $title, $cmd) {
    Write-Host "[reproduce_all $n] $title"
    & bash -c $cmd
}

Run-Step "1/10" "Environment probe" "python -c 'import json,torch;print(json.dumps(dict(torch=torch.__version__,cuda=torch.cuda.is_available()),indent=2))'"
Run-Step "2/10" "Data + model check" "ls $Root/upstream/RF-Diffusion/dataset/wifi/cond | wc -l"
Run-Step "3/10" "Wi-Fi smoke test" "python scripts/run_experiment.py --task wifi --mode smoke-test --num-samples 2"
Run-Step "4/10" "Wi-Fi official"   "python scripts/run_official_wifi.py"
Run-Step "5/10" "5G official"      "python scripts/run_official_mimo.py"
Run-Step "6/10" "Small-scale train" "python scripts/run_small_train.py"
Run-Step "7/10" "Efficiency sweep" "python scripts/run_efficiency.py"
Run-Step "8/10" "Aggregate"        "python scripts/aggregate_metrics.py"
Run-Step "9/10" "Figures"          "python scripts/plot_results.py"
Run-Step "10/10" "LaTeX"           "cd report && bash build.sh"
Write-Host "[reproduce_all] DONE"