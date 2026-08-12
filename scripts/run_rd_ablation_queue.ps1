param(
    [int]$BatchSize = 2,
    [int]$PollSeconds = 15
)

$ErrorActionPreference = 'Stop'
if ($BatchSize -lt 2) {
    throw 'BatchSize must be at least 2 for the RD ablation protocol.'
}
$root = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\Surfa\AppData\Local\Programs\Python\Python39\python.exe'
$dataset = @(
    Get-ChildItem -LiteralPath 'K:\' -Directory -ErrorAction Stop |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -ErrorAction SilentlyContinue } |
        ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -Filter 'MAT' -ErrorAction SilentlyContinue }
) | Select-Object -First 1 | ForEach-Object { $_.Parent.FullName }
if ([string]::IsNullOrWhiteSpace($dataset)) { throw 'Could not locate a dataset directory containing MAT under K drive' }
$artifacts = Join-Path $root 'artifacts'

# One seed, one run per condition. A0 already exists as the baseline.
$experiments = @(
    @{ id='A1'; name='ablation_A1_vr90_w180_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','180','--resampling','db_linear','--normalization','global_z','--input-mode','rd') },
    @{ id='A2'; name='ablation_A2_vr90_w720_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','720','--resampling','db_linear','--normalization','global_z','--input-mode','rd') },
    @{ id='A3'; name='ablation_A3_vrwide_w512_mask_seed42'; args=@('--velocity-min','-240','--velocity-max','238','--target-width','512','--resampling','db_linear','--normalization','global_z','--input-mode','rd_mask') },
    @{ id='A4'; name='ablation_A4_vrwide_w512_seed42'; args=@('--velocity-min','-240','--velocity-max','238','--target-width','512','--resampling','db_linear','--normalization','global_z','--input-mode','rd') },
    @{ id='B1'; name='ablation_B1_power_linear_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','power_linear','--normalization','global_z','--input-mode','rd') },
    @{ id='B2'; name='ablation_B2_area_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','area','--normalization','global_z','--input-mode','rd') },
    @{ id='B3'; name='ablation_B3_db_linear_mask_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','global_z','--input-mode','rd_mask') },
    @{ id='C1'; name='ablation_C1_frame_z_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','frame_z','--input-mode','rd') },
    @{ id='C2'; name='ablation_C2_frame_robust_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','frame_robust','--input-mode','rd') },
    @{ id='C3'; name='ablation_C3_minmax_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','minmax','--input-mode','rd') },
    @{ id='C4'; name='ablation_C4_clip_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','clip','--input-mode','rd') },
    @{ id='D1'; name='ablation_D1_mask_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','global_z','--input-mode','rd_mask') },
    @{ id='D2'; name='ablation_D2_peak_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','global_z','--input-mode','rd_peak') },
    @{ id='D3'; name='ablation_D3_background_seed42'; args=@('--velocity-min','-90','--velocity-max','89','--target-width','360','--resampling','db_linear','--normalization','global_z','--input-mode','rd_background') }
)

function Get-TrainingProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'radar_rd.train' })
}

function Start-Ablation($experiment) {
    $output = Join-Path $artifacts $experiment.name
    if (Test-Path $output) {
        $config = Join-Path $output 'config.json'
        $complete = Join-Path $output 'ablation_complete.json'
        if (Test-Path $complete) { return $null }
        throw "Output directory exists but is incomplete: $output"
    }
    $stdout = Join-Path $artifacts "$($experiment.name).log"
    $stderr = Join-Path $artifacts "$($experiment.name).err.log"
    $arguments = @('-m','radar_rd.train','--dataset-root',$dataset,'--output-dir',$output,'--epochs','50','--seed','42','--workers','4','--batch-size','128','--max-train-frames-per-trajectory','32','--norm-samples','2048','--patience','10','--learning-rate','0.0003','--weight-decay','0.0001','--skip-test') + $experiment.args
    Write-Host "Starting $($experiment.id): $($experiment.name)" -ForegroundColor Cyan
    Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
}

function Find-ExistingTrainingProcess($experimentName) {
    $needle = [regex]::Escape((Join-Path $artifacts $experimentName))
    $candidate = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'radar_rd\.train' -and $_.CommandLine -match $needle } |
        Select-Object -First 1
    if ($null -eq $candidate) { return $null }
    return Get-Process -Id $candidate.ProcessId -ErrorAction SilentlyContinue
}

# Dynamic scheduling keeps at least two experiments active whenever pending
# conditions remain. It also resumes safely after the monitor is restarted.
$pending = [System.Collections.Generic.Queue[object]]::new()
foreach ($experiment in $experiments) { [void]$pending.Enqueue($experiment) }
$active = [System.Collections.ArrayList]::new()

# Remove completed conditions and reattach already-running children before
# scheduling new work. This is important when resuming after a queue crash.
$initialPending = [System.Collections.Generic.Queue[object]]::new()
while ($pending.Count -gt 0) {
    $experiment = $pending.Dequeue()
    $output = Join-Path $artifacts $experiment.name
    if (Test-Path (Join-Path $output 'ablation_complete.json')) { continue }
    $existing = Find-ExistingTrainingProcess $experiment.name
    if ($null -ne $existing) {
        [void]$active.Add([pscustomobject]@{ Experiment = $experiment; Process = $existing })
        Write-Host "Reattached $($experiment.id): $($experiment.name)" -ForegroundColor Yellow
    } else {
        [void]$initialPending.Enqueue($experiment)
    }
}
$pending = $initialPending

while ($pending.Count -gt 0 -or $active.Count -gt 0) {

    while ($active.Count -lt $BatchSize -and $pending.Count -gt 0) {
        $experiment = $pending.Dequeue()
        $process = Start-Ablation $experiment
        if ($null -ne $process) {
            [void]$active.Add([pscustomobject]@{ Experiment = $experiment; Process = $process })
        }
    }

    if ($active.Count -eq 0) { continue }
    Start-Sleep -Seconds $PollSeconds
    foreach ($entry in @($active)) {
        $process = $entry.Process
        if (-not $process.HasExited) { continue }
        # Start-Process can expose a null ExitCode after the child has already
        # exited. Treat that as a normal completion; only explicit non-zero is
        # a failed experiment.
        if ($null -ne $process.ExitCode -and [int]$process.ExitCode -ne 0) {
            throw "Ablation failed: $($entry.Experiment.name), exit code $($process.ExitCode)"
        }
        $completion = Join-Path (Join-Path $artifacts $entry.Experiment.name) 'ablation_complete.json'
        if (-not (Test-Path $completion)) {
            throw "Ablation stopped without completion marker: $($entry.Experiment.name)"
        }
        Write-Host "Experiment complete: $($entry.Experiment.name)" -ForegroundColor Green
        [void]$active.Remove($entry)
    }
    # Refill immediately in the same polling cycle so the active count does
    # not remain below the requested concurrency between iterations.
    while ($active.Count -lt $BatchSize -and $pending.Count -gt 0) {
        $experiment = $pending.Dequeue()
        $process = Start-Ablation $experiment
        if ($null -ne $process) {
            [void]$active.Add([pscustomobject]@{ Experiment = $experiment; Process = $process })
        }
    }
    Write-Host "Ablation active: $($active.Count); pending: $($pending.Count)" -ForegroundColor DarkCyan
}
Write-Host 'RD ablation queue complete.' -ForegroundColor Green
