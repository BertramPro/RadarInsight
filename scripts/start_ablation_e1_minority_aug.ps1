$ErrorActionPreference = 'Stop'

$python = 'C:\Users\Surfa\AppData\Local\Programs\Python\Python39\python.exe'
$root = Split-Path -Parent $PSScriptRoot
$output = Join-Path $root 'artifacts\ablation_E1_minority_aug_seed42_retry'
$log = Join-Path $root 'artifacts\ablation_E1_minority_aug_seed42_retry.log'
$errorLog = Join-Path $root 'artifacts\ablation_E1_minority_aug_seed42_retry.err.log'
$dataset = Get-ChildItem -LiteralPath 'K:\' -Directory |
    ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue } |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'MAT') } |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $dataset) { throw 'No dataset directory containing MAT was found under K:\.' }

if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $log) -or (Test-Path -LiteralPath $errorLog)) {
    throw "Experiment output or log already exists: $output"
}

$arguments = @(
    (Join-Path $root 'radar_rd\train.py'), '--dataset-root', $dataset, '--output-dir', $output,
    '--epochs', '50', '--batch-size', '128', '--workers', '4',
    '--max-train-frames-per-trajectory', '32', '--norm-samples', '2048',
    '--learning-rate', '0.05', '--weight-decay', '0.0001', '--patience', '10', '--seed', '42',
    '--velocity-min', '-90', '--velocity-max', '89', '--target-width', '360',
    '--resampling', 'db_linear', '--normalization', 'global_z', '--input-mode', 'rd',
    '--augmentation', 'minority_rd', '--skip-test'
)

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError $errorLog -WindowStyle Hidden -PassThru
Write-Host "Started ablation_E1_minority_aug_seed42 (PID $($process.Id))." -ForegroundColor Cyan
