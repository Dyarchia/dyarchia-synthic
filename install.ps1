$ErrorActionPreference = "Stop"
$repo = "Dyarchia/synthic"
$name = "synthic"
$target = Join-Path $env:LOCALAPPDATA "Programs\$name"
$release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest"
$asset = $release.assets | Where-Object { $_.name -like "$name-*-windows-x64.zip" } | Select-Object -First 1
if (-not $asset) { throw "Release $($release.tag_name) has no Windows build" }
$sum = $release.assets | Where-Object { $_.name -eq "$($asset.name).sha256" } | Select-Object -First 1
if (-not $sum) { throw "Release $($release.tag_name) has no checksum for $($asset.name)" }
$zip = Join-Path $env:TEMP $asset.name
Invoke-WebRequest $asset.browser_download_url -OutFile $zip
$expected = ((Invoke-WebRequest $sum.browser_download_url).Content | Out-String).Trim().Split(" ")[0]
$actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) {
    Remove-Item $zip
    throw "Checksum mismatch for $($asset.name)"
}
if (Test-Path $target) { Remove-Item $target -Recurse -Force }
Expand-Archive $zip -DestinationPath $target
Remove-Item $zip
$path = [Environment]::GetEnvironmentVariable("Path", "User")
if (($path -split ";") -notcontains $target) {
    [Environment]::SetEnvironmentVariable("Path", "$path;$target".Trim(";"), "User")
}
Write-Host "$name $($release.tag_name) installed in $target. Open a new terminal to use it."
