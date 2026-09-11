<#
.SYNOPSIS
Demo sinh van ban: dat Hyena canh Transformer tren cung mot prompt.

.DESCRIPTION
Bao boc `python -m hyena_study.generate` de luc thuyet trinh chi can go mot dong
ngan. Hai mo hinh duoc huan luyen tren cung ngan sach token, cung sieu tham so,
chi khac nhau o toan tu tron token.

CAN CO TRUOC: results/DEMO_vi_HHHH_s0.pt va results/DEMO_vi_AAAA_s0.pt
(sinh boi notebooks/kaggle_demo_generate.ipynb tren Kaggle GPU).

.EXAMPLE
.\demo.ps1
.EXAMPLE
.\demo.ps1 "Viet Nam la mot quoc gia nam o"
.EXAMPLE
.\demo.ps1 "Ha Noi la thu do cua" -Tokens 60 -Seed 3
.EXAMPLE
.\demo.ps1 "Mo hinh ngon ngu" -HyenaOnly
.EXAMPLE
.\demo.ps1 -Ui          # mo giao dien web tren trinh duyet (dung cai nay de demo)
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Prompt = "Trường Đại học Công nghệ Thông tin",

    [ValidateRange(1, 512)]
    [int]$Tokens = 40,

    # 0 hoac nho hon = lay argmax, tat dinh, khong phu thuoc seed
    [ValidateRange(0.0, 5.0)]
    [double]$Temperature = 0.9,

    [ValidateRange(0, 16000)]
    [int]$TopK = 40,

    [int]$Seed = 0,

    # Mo GIAO DIEN WEB thay vi in ra console. Dung cai nay khi trinh bay.
    [switch]$Ui,

    # Cong cho giao dien web, doi khi 8000 dang bi chiem
    [int]$Port = 8000,

    # Chi chay Hyena, bo nhanh Transformer doi chung
    [switch]$HyenaOnly,

    # Cho phep mo hinh nha ra token khong hien thi duoc (zero-width, dau phu)
    [switch]$KeepInvisible
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$hyena = "results/DEMO_vi_HHHH_s0.pt"
$trans = "results/DEMO_vi_AAAA_s0.pt"

foreach ($f in @($hyena, $trans)) {
    if (-not (Test-Path -LiteralPath $f)) {
        if ($f -eq $trans -and $HyenaOnly) { continue }
        Write-Host ""
        Write-Host "KHONG THAY CHECKPOINT: $f" -ForegroundColor Red
        Write-Host "Chay notebooks/kaggle_demo_generate.ipynb tren Kaggle GPU," -ForegroundColor Yellow
        Write-Host "tai demo_checkpoints.zip ve roi giai nen vao thu muc results/." -ForegroundColor Yellow
        Write-Host ""
        exit 1
    }
}

$env:PYTHONIOENCODING = "utf-8"

if ($Ui) {
    $uiArgs = @("-m", "hyena_study.serve", "--ckpt", $hyena, "--port", $Port, "--device", "cpu")
    if ($HyenaOnly) { $uiArgs += @("--compare", "") } else { $uiArgs += @("--compare", $trans) }
    Write-Host ""
    Write-Host "Dang nap mo hinh va mo giao dien tren http://127.0.0.1:$Port/ ..." -ForegroundColor Cyan
    Write-Host "Nhan Ctrl+C trong cua so nay de dung." -ForegroundColor DarkGray
    Write-Host ""
    & python @uiArgs
    exit $LASTEXITCODE
}

$cmdArgs = @(
    "-m", "hyena_study.generate",
    "--ckpt", $hyena,
    "--prompt", $Prompt,
    "--max_new_tokens", $Tokens,
    "--temperature", $Temperature,
    "--top_k", $TopK,
    "--seed", $Seed,
    "--device", "cpu"
)
if (-not $HyenaOnly)   { $cmdArgs += @("--compare", $trans) }
if ($KeepInvisible)    { $cmdArgs += "--keep_invisible" }

# Python tren Windows nhan argv dang Unicode nen prompt tieng Viet di qua nguyen ven.
# PYTHONIOENCODING o tren chi bao dam phan IN RA man hinh khong hong voi console cu.
& python @cmdArgs
exit $LASTEXITCODE
