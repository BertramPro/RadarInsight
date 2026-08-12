$py = 'C:\Users\Surfa\AppData\Local\Programs\Python\Python39\python.exe'
$server = Join-Path $PSScriptRoot 'training_monitor_server.py'
Start-Process -FilePath $py -ArgumentList @($server) -WorkingDirectory (Split-Path -Parent $PSScriptRoot) -WindowStyle Hidden
Start-Sleep -Seconds 1
Write-Host 'Training monitor is available on this computer at http://127.0.0.1:8765' -ForegroundColor Cyan
Write-Host 'For other devices, use this computer''s LAN address with port 8765.' -ForegroundColor Cyan
Start-Process 'http://127.0.0.1:8765'
