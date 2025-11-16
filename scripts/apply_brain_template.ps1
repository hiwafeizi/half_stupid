# Applies a standardized section template to every .txt file under research/brain
# Sections are separated by two blank lines.

$root = Join-Path $PSScriptRoot "..\research\brain" | Resolve-Path
$template = @"
Functionality:
- 


What affects it:
- 


How it affects other parts:
- 


Explanation:
- 


Implementation suggestions (code):
- 
"@

Get-ChildItem -Path $root -Recurse -File -Filter *.txt | ForEach-Object {
  $path = $_.FullName
  $content = Get-Content -Raw -LiteralPath $path -ErrorAction SilentlyContinue
  if ($null -eq $content) { $content = "" }
  # If the template already exists (detect by section header anywhere), skip
  if ($content -match "(?m)^Functionality:") {
    return
  }
  $sep = if ([string]::IsNullOrWhiteSpace($content)) { "" } else { "`r`n`r`n" }
  $newContent = $content + $sep + $template
  # Write back as UTF8 (BOM ok for notes)
  Set-Content -LiteralPath $path -Value $newContent -Encoding utf8
  Write-Host "Updated: $path"
}
