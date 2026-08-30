param(
    [switch]$OpenEpicLauncher
)

$ErrorActionPreference = "Stop"
$LogPath = Join-Path $env:LOCALAPPDATA "Temp\everest-unreal-prereqs.log"
$MarkerPath = Join-Path $env:LOCALAPPDATA "Temp\everest-unreal-prereqs.done"
Remove-Item $MarkerPath -Force -ErrorAction SilentlyContinue
"Everest Unreal prerequisites started: $(Get-Date -Format o)" | Set-Content $LogPath

function Invoke-WingetInstall {
    param(
        [Parameter(Mandatory = $true)][string]$Id,
        [string]$Override = ""
    )

    $installed = winget list --id $Id --exact --accept-source-agreements 2>$null | Out-String
    if ($installed -match [regex]::Escape($Id)) {
        "Already installed: $Id" | Tee-Object -FilePath $LogPath -Append
        return
    }

    $args = @(
        "install", "--id", $Id, "--exact", "--silent",
        "--accept-source-agreements", "--accept-package-agreements",
        "--disable-interactivity"
    )
    if ($Override) {
        $args += @("--override", $Override)
    }

    "Installing: $Id" | Tee-Object -FilePath $LogPath -Append
    & winget @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed for $Id with exit code $LASTEXITCODE"
    }
}

try {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "winget.exe is not available"
    }

    Invoke-WingetInstall -Id "EpicGames.EpicGamesLauncher"
    Invoke-WingetInstall -Id "Microsoft.VisualStudio.2022.BuildTools" -Override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vcPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vcPath) {
            "VC toolchain ready: $vcPath" | Tee-Object -FilePath $LogPath -Append
        }
    }

    "SUCCESS $(Get-Date -Format o)" | Set-Content $MarkerPath
    "Everest Unreal prerequisites complete." | Tee-Object -FilePath $LogPath -Append

    if ($OpenEpicLauncher) {
        Start-Process "com.epicgames.launcher://ue/library"
    }
}
catch {
    "FAILED: $($_.Exception.Message)" | Tee-Object -FilePath $LogPath -Append
    throw
}
