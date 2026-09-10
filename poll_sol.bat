@echo off
REM Hourly poll for the SOL/USD forward test. Registered as the scheduled task
REM "quantlab-sol-trend"; see README, "Routing the orders to a broker".
REM
REM Hourly rather than once at 00:00 UTC on purpose: the daily bar closes at
REM 17:00 Pacific and that is the only moment this trades all day, so a missed
REM poll is a missed trade and a hole in the record. Hourly is self-healing --
REM if one run dies, the next hour picks the bar up.
REM
REM Safe to run by hand at any time. A poll with no new closed bar does nothing.

cd /d "C:\Users\Redux\tradingview-mcp"

echo. >> "paper_runs\sol-trend-20260906\poll.log"
echo ---- %DATE% %TIME% ---- >> "paper_runs\sol-trend-20260906\poll.log"
"C:\Python314\python.exe" paper.py poll --id sol-trend-20260906 >> "paper_runs\sol-trend-20260906\poll.log" 2>&1

exit /b %ERRORLEVEL%
