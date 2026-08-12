param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [Parameter(Mandatory = $true)]
    [string]$ErrorLogPath,
    [int]$TotalEpochs = 50
)

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = New-Object System.Windows.Forms.Form
$form.Text = "R1 RD CNN Training"
$form.Width = 760
$form.Height = 650
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedSingle"
$form.MaximizeBox = $false
$form.BackColor = [System.Drawing.Color]::FromArgb(245, 247, 250)

$title = New-Object System.Windows.Forms.Label
$title.Text = "R1 RD CNN | Physical-Vr Resampling | 31 x 360"
$title.Font = New-Object System.Drawing.Font("Segoe UI", 14, [System.Drawing.FontStyle]::Bold)
$title.ForeColor = [System.Drawing.Color]::FromArgb(30, 41, 59)
$title.AutoSize = $true
$title.Location = New-Object System.Drawing.Point(24, 20)
$form.Controls.Add($title)

$epochLabel = New-Object System.Windows.Forms.Label
$epochLabel.Text = "Preparing..."
$epochLabel.Font = New-Object System.Drawing.Font("Segoe UI", 12)
$epochLabel.AutoSize = $true
$epochLabel.Location = New-Object System.Drawing.Point(26, 62)
$form.Controls.Add($epochLabel)

$progress = New-Object System.Windows.Forms.ProgressBar
$progress.Minimum = 0
$progress.Maximum = 100
$progress.Value = 0
$progress.Style = [System.Windows.Forms.ProgressBarStyle]::Continuous
$progress.Width = 700
$progress.Height = 28
$progress.Location = New-Object System.Drawing.Point(26, 94)
$form.Controls.Add($progress)

$metricLabel = New-Object System.Windows.Forms.Label
$metricLabel.Text = "Validation loss: --    |    Macro-F1: --"
$metricLabel.Font = New-Object System.Drawing.Font("Segoe UI", 11)
$metricLabel.AutoSize = $true
$metricLabel.Location = New-Object System.Drawing.Point(26, 132)
$form.Controls.Add($metricLabel)

$statusLabel = New-Object System.Windows.Forms.Label
$statusLabel.Text = "Process: $ProcessId"
$statusLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
$statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(71, 85, 105)
$statusLabel.AutoSize = $true
$statusLabel.Location = New-Object System.Drawing.Point(26, 163)
$form.Controls.Add($statusLabel)

$configBox = New-Object System.Windows.Forms.GroupBox
$configBox.Text = "Training configuration"
$configBox.Font = New-Object System.Drawing.Font("Segoe UI", 9, [System.Drawing.FontStyle]::Bold)
$configBox.Width = 700
$configBox.Height = 82
$configBox.Location = New-Object System.Drawing.Point(26, 188)
$form.Controls.Add($configBox)

$configLabel = New-Object System.Windows.Forms.Label
$configLabel.Font = New-Object System.Drawing.Font("Consolas", 9)
$configLabel.AutoSize = $false
$configLabel.Width = 670
$configLabel.Height = 55
$configLabel.Location = New-Object System.Drawing.Point(12, 20)
$configLabel.Text = "Loading config..."
$configBox.Controls.Add($configLabel)

$historyLabel = New-Object System.Windows.Forms.Label
$historyLabel.Text = "Epoch history"
$historyLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10, [System.Drawing.FontStyle]::Bold)
$historyLabel.AutoSize = $true
$historyLabel.Location = New-Object System.Drawing.Point(26, 280)
$form.Controls.Add($historyLabel)

$history = New-Object System.Windows.Forms.ListView
$history.View = [System.Windows.Forms.View]::Details
$history.FullRowSelect = $true
$history.GridLines = $true
$history.HideSelection = $false
$history.Width = 700
$history.Height = 300
$history.Location = New-Object System.Drawing.Point(26, 308)
[void]$history.Columns.Add("Epoch", 70)
[void]$history.Columns.Add("Train loss", 120)
[void]$history.Columns.Add("Val loss", 120)
[void]$history.Columns.Add("Frame accuracy", 150)
[void]$history.Columns.Add("Val Macro-F1", 150)
[void]$history.Columns.Add("Val accuracy", 150)
$form.Controls.Add($history)

$seenEpochs = @{}
$logParent = Split-Path -Parent $LogPath
$logStem = [System.IO.Path]::GetFileNameWithoutExtension($LogPath)
$progressPath = Join-Path (Join-Path $logParent $logStem) "progress.json"

function Read-Metrics {
    $items = @()
    if (-not (Test-Path -LiteralPath $LogPath)) {
        return $items
    }
    $lines = Get-Content -LiteralPath $LogPath -Encoding utf8 -Tail 200 -ErrorAction SilentlyContinue
    foreach ($line in $lines) {
        if ($line -match '^\{"epoch"') {
            try {
                $items += ($line | ConvertFrom-Json)
            }
            catch {
            }
        }
    }
    return $items
}

function Read-Config {
    $logParent = Split-Path -Parent $LogPath
    $logStem = [System.IO.Path]::GetFileNameWithoutExtension($LogPath)
    $candidates = @(
        (Join-Path $logParent "config.json"),
        (Join-Path (Join-Path $logParent $logStem) "config.json")
    )
    $configPath = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ($null -eq $configPath) {
        $configLabel.Text = "Config not found"
        return
    }
    try {
        $config = Get-Content -LiteralPath $configPath -Encoding utf8 -Raw | ConvertFrom-Json
        $configLabel.Text = "LR: $($config.learning_rate)    Batch: $($config.batch_size)    Epochs: $($config.epochs)    Seed: $($config.seed)`r`n" +
            "Weight decay: $($config.weight_decay)    Train frames/trajectory: $($config.max_train_frames_per_trajectory)`r`n" +
            "Vr: [$($config.velocity_preprocessing.common_interval_mps[0]), $($config.velocity_preprocessing.common_interval_mps[1])] m/s -> $($config.velocity_preprocessing.target_width) columns`r`n" +
            "Normalization: mean=$([math]::Round($config.normalization_mean, 3)), std=$([math]::Round($config.normalization_std, 3))"
    }
    catch {
        $configLabel.Text = "Config unavailable: $($_.Exception.Message)"
    }
}

function Update-Window {
    if (Test-Path -LiteralPath $progressPath) {
        try {
            $live = Get-Content -LiteralPath $progressPath -Encoding utf8 -Raw | ConvertFrom-Json
            $batchPercent = 0.0
            if ($live.total_batches -gt 0) {
                $batchPercent = 100.0 * [double]$live.batch / [double]$live.total_batches
            }
            $progress.Value = [Math]::Min(100, [Math]::Max(0, [int][Math]::Round($batchPercent)))
            if ($live.phase -eq "train") {
                $epochLabel.Text = "Epoch $($live.epoch) / $TotalEpochs"
                $metricLabel.Text = "Batch $($live.batch) / $($live.total_batches)    |    Train loss: {1:N4}    |    Frame accuracy: {2:P1}" -f `
                    ($live.batch / [Math]::Max(1, $live.total_batches)), [double]$live.train_loss, [double]$live.train_frame_accuracy
            }
            elseif ($live.phase -eq "validation") {
                $progress.Value = 100
                $epochLabel.Text = "Epoch $($live.epoch) / $TotalEpochs    |    Validation"
                $metricLabel.Text = "Validation batch $($live.batch) / $($live.total_batches)    |    Loss: {0:N4}" -f [double]$live.loss
            }
            elseif ($live.phase -eq "testing") {
                $progress.Value = [Math]::Min(100, [Math]::Max(0, [int][Math]::Round(100.0 * $live.batch / [Math]::Max(1, $live.total_batches))))
                $epochLabel.Text = "Testing"
                $metricLabel.Text = "Test batch $($live.batch) / $($live.total_batches)    |    Loss: {0:N4}" -f [double]$live.loss
            }
        }
        catch {
        }
    }
    $metrics = Read-Metrics
    foreach ($metric in $metrics) {
        $epoch = [int]$metric.epoch
        if (-not $seenEpochs.ContainsKey($epoch)) {
            $seenEpochs[$epoch] = $true
            $item = New-Object System.Windows.Forms.ListViewItem($epoch.ToString())
            [void]$item.SubItems.Add(('{0:N4}' -f [double]$metric.train_loss))
            if ($null -eq $metric.val_loss) {
                [void]$item.SubItems.Add("--")
            }
            else {
                [void]$item.SubItems.Add(('{0:N4}' -f [double]$metric.val_loss))
            }
            [void]$item.SubItems.Add(('{0:P1}' -f [double]$metric.train_frame_accuracy))
            [void]$item.SubItems.Add(('{0:P1}' -f [double]$metric.val_trajectory_macro_f1))
            [void]$item.SubItems.Add(('{0:P1}' -f [double]$metric.val_trajectory_accuracy))
            [void]$history.Items.Add($item)
        }
    }
    if ($metrics.Count -gt 0) {
        $latest = $metrics | Sort-Object epoch | Select-Object -Last 1
        $currentEpoch = [Math]::Min($TotalEpochs, [Math]::Max(0, [int]$latest.epoch))
        if (-not (Test-Path -LiteralPath $progressPath)) {
            $progress.Value = 100
            $epochLabel.Text = "Epoch $currentEpoch / $TotalEpochs"
            if ($null -eq $latest.val_loss) {
                $metricLabel.Text = "Validation loss: --    |    Macro-F1: {0:P1}    |    Accuracy: {1:P1}" -f `
                    [double]$latest.val_trajectory_macro_f1, [double]$latest.val_trajectory_accuracy
            }
            else {
                $metricLabel.Text = "Validation loss: {0:N4}    |    Macro-F1: {1:P1}    |    Accuracy: {2:P1}" -f `
                    [double]$latest.val_loss, [double]$latest.val_trajectory_macro_f1, [double]$latest.val_trajectory_accuracy
            }
        }
    }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        $statusLabel.Text = "Training process ended"
        $statusLabel.ForeColor = [System.Drawing.Color]::FromArgb(22, 101, 52)
        $timer.Stop()
        return
    }
    $statusLabel.Text = "Process $ProcessId running | CPU {0:N1}s | Memory {1:N2} GB" -f `
        [double]$process.CPU, ($process.WorkingSet64 / 1GB)
}

$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 1500
$timer.Add_Tick({ Read-Config; Update-Window })
$form.Add_Shown({ Read-Config; Update-Window; $timer.Start() })
$form.Add_FormClosed({ $timer.Stop(); $timer.Dispose() })
[void]$form.ShowDialog()
