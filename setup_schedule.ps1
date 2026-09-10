# Registers the hourly poll for the SOL/USD forward test.
#
# Run once:   powershell -ExecutionPolicy Bypass -File .\setup_schedule.ps1
# Remove it:  Unregister-ScheduledTask -TaskName "quantlab-sol-trend" -Confirm:$false
#
# Why hourly and not once a day: the daily bar closes at 00:00 UTC (17:00 Pacific)
# and that is the only moment this strategy trades. A single daily task that misses
# -- asleep, no network, reboot -- is a missed trade and a hole in a record whose
# entire value is being continuous. Hourly is self-healing: a poll with no new
# closed bar does nothing, so the extra runs cost a few API calls and nothing else.

$ErrorActionPreference = "Stop"

$name    = "quantlab-sol-trend"
$bat     = "C:\Users\Redux\tradingview-mcp\poll_sol.bat"
$workdir = "C:\Users\Redux\tradingview-mcp"

if (-not (Test-Path $bat)) { throw "missing $bat" }

# replace any previous version of this task
try { Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop } catch {}

$action = New-ScheduledTaskAction -Execute $bat -WorkingDirectory $workdir

# start at the top of the next hour, then every hour forever
$start   = (Get-Date).Date.AddHours((Get-Date).Hour + 1)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Hours 1)

# StartWhenAvailable catches up a run the machine slept through, which is the whole
# point. The battery flags matter on a laptop -- without them Windows silently
# skips the task on battery and the record just stops.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew

$user = "$env:USERDOMAIN\$env:USERNAME"

# S4U runs with no stored password AND with no console window. It needs the "log on
# as a batch job" right, which not every account has, so fall back to Interactive --
# that one only runs while you are logged in and flashes a console once an hour.
try {
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description "Hourly poll of the SOL/USD trend_filter forward test (quantlab)." | Out-Null
    Write-Host "registered '$name' (S4U: runs hidden, and while logged out)" -ForegroundColor Green
} catch {
    Write-Host "S4U unavailable ($($_.Exception.Message.Trim())) - falling back to Interactive" -ForegroundColor Yellow
    $principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger `
        -Settings $settings -Principal $principal `
        -Description "Hourly poll of the SOL/USD trend_filter forward test (quantlab)." | Out-Null
    Write-Host "registered '$name' (Interactive: only runs while logged in)" -ForegroundColor Green
}

Get-ScheduledTask -TaskName $name | Select-Object TaskName, State | Format-Table -AutoSize
Write-Host "first run: $start   then hourly"
Write-Host "test it now:  Start-ScheduledTask -TaskName '$name'"
Write-Host "watch it:     Get-Content '$workdir\paper_runs\sol-trend-20260906\poll.log' -Tail 20 -Wait"
