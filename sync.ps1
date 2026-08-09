# Sync helper for labellers.
#
# Run BEFORE you start labelling and AFTER you finish, so your labels land in
# the shared repo without merge conflicts. Every labeller only writes their own
# per-labeller files, so a rebase-based pull merges everyone's work cleanly.
#
#   ./sync.ps1
#
param(
    [string]$Message = "labels: sync"
)

$ErrorActionPreference = "Stop"

Write-Host "Pulling latest changes..." -ForegroundColor Cyan
git pull --rebase --autostash

git add labels snapshots

if (git status --porcelain -- labels snapshots) {
    Write-Host "Committing your labels and snapshots..." -ForegroundColor Cyan
    git commit -m $Message
    Write-Host "Pushing..." -ForegroundColor Cyan
    git push
    Write-Host "Done. Your labels are pushed." -ForegroundColor Green
} else {
    Write-Host "Nothing to commit. You are up to date." -ForegroundColor Green
}
