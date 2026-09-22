param([int]$Port = 8766, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$viewerRoot = $PSScriptRoot
$viewerUrl = "http://127.0.0.1:$Port/"
$existingViewer = $null
try {
    $existingViewer = Invoke-RestMethod -Uri ($viewerUrl + 'api/health') -TimeoutSec 2
} catch { }
if ($null -ne $existingViewer) {
    if ($existingViewer.app -eq 'GTM local experiment viewer') { if (-not $NoBrowser) { Start-Process $viewerUrl }; Write-Output $viewerUrl; exit 0 }
    throw "Port $Port is in use by another application. Choose a different -Port."
}
$viewerPython = Join-Path $env:USERPROFILE '.pyenv/pyenv-win/versions/3.11.9/python.exe'
if (-not (Test-Path -LiteralPath $viewerPython)) {
    $viewerPython = (Get-Command python.exe -ErrorAction Stop).Source
}
$viewerScript = Join-Path $viewerRoot 'tools/GTM_Model0_local_viewer.py'
$viewerLogs = Join-Path $viewerRoot 'outputs/GTM_viewer_logs'
New-Item -ItemType Directory -Path $viewerLogs -Force | Out-Null
$viewerArguments = @(('"' + $viewerScript + '"'), '--port', $Port)
$viewerProcess = Start-Process -FilePath $viewerPython -ArgumentList $viewerArguments -WorkingDirectory $viewerRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $viewerLogs 'server.stdout.log') -RedirectStandardError (Join-Path $viewerLogs 'server.stderr.log')
$viewerProcess.Id | Set-Content -LiteralPath (Join-Path $viewerLogs 'server.pid')
for ($viewerTry=0; $viewerTry -lt 30; $viewerTry++) {
    try { $viewerHealth = Invoke-RestMethod -Uri ($viewerUrl + 'api/health') -TimeoutSec 1; if($viewerHealth.app -eq 'GTM local experiment viewer'){ if (-not $NoBrowser) { Start-Process $viewerUrl }; Write-Output $viewerUrl; exit 0 } } catch { }
    Start-Sleep -Milliseconds 200
}
throw 'Viewer did not start. Check outputs/GTM_viewer_logs/server.stderr.log.'
