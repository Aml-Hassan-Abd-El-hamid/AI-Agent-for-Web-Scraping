<#
.SYNOPSIS
    Downloads and arranges the SWDE (Structured Web Data Extraction) dataset so
    that `swde_benchmark.py prepare --swde-root <OutDir>` works directly.

.DESCRIPTION
    The original SWDE host (CodePlex) is offline. This script pulls the official
    Wayback Machine backup of swde.zip that Microsoft's MarkupLM repo points to,
    extracts it, then normalizes the contents into:

        <OutDir>/sourceCode/<vertical>/<vertical>-<site>(NNNN)/*.htm
        <OutDir>/groundtruth/<vertical>/<vertical>-<site>-<attr>.txt

    Resumable download (curl.exe), safe to re-run.

.EXAMPLE
    ./get_swde.ps1
    python swde_benchmark.py prepare --swde-root swde_data
#>
[CmdletBinding()]
param(
    [string]$OutDir  = "swde_data",
    [string]$WorkDir = "swde_download",
    [string]$Url     = "http://web.archive.org/web/20210630013015/https://codeplexarchive.blob.core.windows.net/archive/projects/swde/swde.zip",
    [switch]$KeepZip
)

$ErrorActionPreference = "Stop"
$verticals = @("auto","book","camera","job","movie","nbaplayer","restaurant","university")

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$zipPath = Join-Path $WorkDir "swde.zip"

# ── 1. Download (resumable) ────────────────────────────────────────────
Write-Host "==> Downloading swde.zip (this is large, ~900 MB) ..." -ForegroundColor Cyan
Write-Host "    $Url"
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if ($curl) {
    # -C - resumes a partial file; -L follows the Wayback redirect.
    & curl.exe -L -C - --retry 5 --retry-delay 5 -o $zipPath $Url
    if ($LASTEXITCODE -ne 0) { throw "curl download failed (exit $LASTEXITCODE)." }
} else {
    Write-Host "    curl.exe not found; falling back to BITS (no resume UI)."
    Start-BitsTransfer -Source $Url -Destination $zipPath
}

$sizeMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
Write-Host "==> Downloaded: $zipPath ($sizeMB MB)" -ForegroundColor Green
if ($sizeMB -lt 100) {
    throw "File looks too small ($sizeMB MB). The download may have failed or " +
          "returned an error page. Delete $zipPath and retry, or open the URL " +
          "in a browser."
}

# ── 2. Extract the outer archive (and any inner .zip) ──────────────────
$extractDir = Join-Path $WorkDir "extracted"
if (Test-Path $extractDir) { Remove-Item -Recurse -Force $extractDir }
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Write-Host "==> Extracting outer archive ..." -ForegroundColor Cyan
try {
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
} catch {
    Write-Host "    Expand-Archive failed; using .NET ZipFile fallback."
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory(
        (Resolve-Path $zipPath), (Resolve-Path $extractDir))
}

# The CodePlex export nests inner .zip archives (sourceCode.zip, downloadWiki.zip).
# Unpack them so the per-vertical payloads become visible.
Get-ChildItem -Path $extractDir -Recurse -Filter *.zip | ForEach-Object {
    Write-Host "    Unpacking inner archive: $($_.Name)"
    try { Expand-Archive -Path $_.FullName -DestinationPath $_.DirectoryName -Force }
    catch {}
}

# ── 3. Extract the dataset into <OutDir>/sourceCode + groundtruth ──────
# SWDE actually ships as per-vertical .7z archives (auto.7z ... university.7z)
# plus groundtruth.7z inside the CodePlex export. Handle that; fall back to
# plain-folder mirrors that already ship the verticals uncompressed.
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$destSource = Join-Path $OutDir "sourceCode"
$destGt     = Join-Path $OutDir "groundtruth"

$all7z  = @(Get-ChildItem -Path $extractDir -Recurse -Filter *.7z -ErrorAction SilentlyContinue)
$vert7z = @($all7z | Where-Object { $verticals -contains $_.BaseName })
$gt7z   = @($all7z | Where-Object { $_.BaseName -eq 'groundtruth' } | Select-Object -First 1)

if ($vert7z.Count -ge 1) {
    Write-Host "==> Found $($vert7z.Count) per-vertical .7z archives; extracting ..." -ForegroundColor Cyan

    # Choose a .7z extractor: prefer built-in tar (libarchive reads 7z), else 7-Zip.
    & tar.exe -tf $vert7z[0].FullName > $null 2>&1
    $tarOk = ($LASTEXITCODE -eq 0)
    $sevenExe = $null
    foreach ($cand in @('7z.exe','7za.exe',
                        'C:\Program Files\7-Zip\7z.exe',
                        'C:\Program Files (x86)\7-Zip\7z.exe')) {
        $found = Get-Command $cand -ErrorAction SilentlyContinue
        if ($found)              { $sevenExe = $found.Source; break }
        elseif (Test-Path $cand) { $sevenExe = $cand; break }
    }
    if (-not $tarOk -and -not $sevenExe) {
        throw "No .7z extractor available: built-in tar cannot read these archives " +
              "and 7-Zip is not installed. Install it (e.g. 'winget install 7zip.7zip') " +
              "and re-run."
    }
    Write-Host "    extractor: $(if ($tarOk) { 'tar (built-in)' } else { $sevenExe })"

    if (Test-Path $destSource) { Remove-Item -Recurse -Force $destSource }
    New-Item -ItemType Directory -Force -Path $destSource | Out-Null

    # Verticals extract into <OutDir>/sourceCode/<vertical>/...
    foreach ($a in $vert7z) {
        Write-Host "    - $($a.BaseName)"
        if ($tarOk) { & tar.exe -xf $a.FullName -C $destSource 2>$null }
        else        { & $sevenExe x $a.FullName "-o$destSource" -y > $null }
    }
    # groundtruth extracts into <OutDir>/groundtruth/...
    if ($gt7z) {
        Write-Host "    - groundtruth"
        if (Test-Path $destGt) { Remove-Item -Recurse -Force $destGt }
        if ($tarOk) { & tar.exe -xf $gt7z.FullName -C $OutDir 2>$null }
        else        { & $sevenExe x $gt7z.FullName "-o$OutDir" -y > $null }
    }
}
else {
    # Fallback: a mirror that already ships uncompressed folders. Detect the
    # sourceCode root (a dir with >=4 vertical subfolders) and groundtruth, move.
    Write-Host "==> No .7z payloads; looking for uncompressed folders ..." -ForegroundColor Cyan
    $sourceRoot = $null
    Get-ChildItem -Path $extractDir -Recurse -Directory | ForEach-Object {
        if ($sourceRoot) { return }
        $childNames = (Get-ChildItem -Path $_.FullName -Directory -ErrorAction SilentlyContinue).Name
        $hits = @($childNames | Where-Object { $verticals -contains $_ })
        if ($hits.Count -ge 4) { $script:sourceRoot = $_.FullName }
    }
    if (-not $sourceRoot) {
        throw "Could not locate SWDE data (neither .7z payloads nor vertical " +
              "folders) under $extractDir. Inspect the extracted files manually."
    }
    $gtRoot = (Get-ChildItem -Path $extractDir -Recurse -Directory -Filter "groundtruth" |
               Select-Object -First 1).FullName

    if (Test-Path $destSource) { Remove-Item -Recurse -Force $destSource }
    Move-Item -Path $sourceRoot -Destination $destSource
    if ($gtRoot -and (Test-Path $gtRoot)) {
        if (Test-Path $destGt) { Remove-Item -Recurse -Force $destGt }
        Move-Item -Path $gtRoot -Destination $destGt
    } elseif (Test-Path (Join-Path $destSource "groundtruth")) {
        Move-Item -Path (Join-Path $destSource "groundtruth") -Destination $destGt
    }
}

if (-not $KeepZip) {
    Write-Host "==> Removing archive + temp extract dir ..."
    Remove-Item -Force $zipPath -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
}

# ── 5. Report ──────────────────────────────────────────────────────────
$vCount = (Get-ChildItem -Path $destSource -Directory |
           Where-Object { $verticals -contains $_.Name }).Count
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " SWDE ready under: $OutDir" -ForegroundColor Green
Write-Host "   sourceCode verticals found: $vCount / 8"
Write-Host "   groundtruth present: $(Test-Path $destGt)"
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next:"
Write-Host "  python swde_benchmark.py prepare --swde-root $OutDir --out-dir swde_prepared"
