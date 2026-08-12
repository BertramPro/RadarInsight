param(
    [int]$ProcessId = 70932,
    [string]$LogPath = "H:\RadarInsight\artifacts\r1_rd_cnn_vr360_seed42.log",
    [string]$ErrorLogPath = "H:\RadarInsight\artifacts\r1_rd_cnn_vr360_seed42.err.log",
    [int]$TotalEpochs = 50
)

$monitor = Join-Path $PSScriptRoot "monitor_r1_rd_cnn.ps1"
$process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Write-Host "Training process $ProcessId is not running. The monitor can still show the saved history." -ForegroundColor Yellow
}
& $monitor -ProcessId $ProcessId -LogPath $LogPath -ErrorLogPath $ErrorLogPath -TotalEpochs $TotalEpochs
