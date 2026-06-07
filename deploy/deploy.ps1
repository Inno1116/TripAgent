param(
    [ValidateSet("init", "up", "start", "update", "bootstrap", "ps", "logs", "down", "stop", "reset")]
    [string]$Action = "up",
    [string]$Service = "api"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$ComposeFile = if ($env:KYURIAGENTS_COMPOSE_FILE) { $env:KYURIAGENTS_COMPOSE_FILE } else { "docker-compose.full.yml" }
$EnvFile = if ($env:KYURIAGENTS_ENV_FILE) { $env:KYURIAGENTS_ENV_FILE } else { "runtime.env" }

function Invoke-Compose {
    docker compose --env-file $EnvFile -f $ComposeFile @args
}

function Ensure-Docker {
    $null = Get-Command docker -ErrorAction Stop
    docker compose version | Out-Null
}

function Ensure-Env {
    if (-not (Test-Path $EnvFile)) {
        Copy-Item runtime.env.example $EnvFile
        Write-Host "Created $EnvFile from runtime.env.example."
        Write-Host "Edit $EnvFile first, then run: .\deploy.ps1 up"
        exit 1
    }
}

function Require-ConfiguredEnv {
    $text = Get-Content $EnvFile -Raw
    $missing = $false
    if ($text -match "(?m)^DASHSCOPE_API_KEY=(replace-me)?$") {
        Write-Error "Please set DASHSCOPE_API_KEY in $EnvFile."
        $missing = $true
    }
    if ($text -match "(?m)^POSTGRES_PASSWORD=(change-this-postgres-password|change-me)?$") {
        Write-Error "Please set a strong POSTGRES_PASSWORD in $EnvFile."
        $missing = $true
    }
    if ($text -notmatch "(?m)^(KYURIAGENTS_API_ADMIN_KEY|DEEPAGENTS_API_ADMIN_KEY)=.+" -or $text -match "(?m)^(KYURIAGENTS_API_ADMIN_KEY|DEEPAGENTS_API_ADMIN_KEY)=(replace-this-admin-key)?$") {
        Write-Error "Please set KYURIAGENTS_API_ADMIN_KEY in $EnvFile."
        $missing = $true
    }
    if ($missing) {
        exit 1
    }
}

Ensure-Docker

switch ($Action) {
    "init" {
        if (Test-Path $EnvFile) {
            Write-Host "$EnvFile already exists."
        } else {
            Copy-Item runtime.env.example $EnvFile
            Write-Host "Created $EnvFile. Edit it before deployment."
        }
    }
    { $_ -in @("up", "start") } {
        Ensure-Env
        Require-ConfiguredEnv
        Invoke-Compose up -d --build
        Invoke-Compose ps
    }
    "update" {
        Ensure-Env
        Require-ConfiguredEnv
        Invoke-Compose up -d --build bootstrap api worker web
        Invoke-Compose ps
    }
    "bootstrap" {
        Ensure-Env
        Require-ConfiguredEnv
        Invoke-Compose up --build bootstrap
    }
    "ps" {
        Ensure-Env
        Invoke-Compose ps
    }
    "logs" {
        Ensure-Env
        Invoke-Compose logs -f $Service
    }
    { $_ -in @("down", "stop") } {
        Ensure-Env
        Invoke-Compose down
    }
    "reset" {
        Ensure-Env
        if ($env:KYURIAGENTS_CONFIRM_RESET -ne "yes") {
            Write-Error "This deletes Docker volumes. Re-run with `$env:KYURIAGENTS_CONFIRM_RESET='yes'; .\deploy.ps1 reset"
        }
        Invoke-Compose down -v
    }
}
