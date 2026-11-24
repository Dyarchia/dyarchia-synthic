param([string]$Python = "python")
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$name = "synthic"
$version = & $Python -c "import $name; print($name.__version__)"
$out = Join-Path $root "build\release"
& $Python -m PyInstaller --noconfirm --clean --onedir --console --name $name `
    --collect-data $name --collect-submodules $name `
    --distpath "$out\dist" --workpath "$out\work" --specpath "$out\spec" (Join-Path $PSScriptRoot "entry.py")
if ($LASTEXITCODE) { throw "PyInstaller failed" }
$zip = Join-Path $out "$name-$version-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "$out\dist\$name\*" -DestinationPath $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Set-Content "$zip.sha256" "$hash  $(Split-Path $zip -Leaf)" -Encoding ascii
Write-Host $zip
