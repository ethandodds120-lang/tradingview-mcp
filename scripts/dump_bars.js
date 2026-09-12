#!/usr/bin/env node
// Page the chart's main series back as far as the feed allows and write every
// bar to a CSV — no LLM, no 500-bar cap, no context cost.
//
// The MCP's `ohlcv` reads the last N of whatever the chart has loaded, and the
// chart lazy-loads ~300 bars at a time. TradingView will serve years of intraday
// history to a paid plan, but only by asking the main series for more data,
// page by page, and waiting for each page to land. That is what
// core/chart.js:setVisibleRange does for 25 pages; this does it until the feed
// says it has no more, then reads the whole series by index.
//
//   node scripts/dump_bars.js OUT.csv [UNTIL_ISO_DATE] [MAX_PAGES]
//
// Set the symbol and timeframe on the chart first (paper.py's TradingView feed,
// or `node src/cli/index.js symbol CME_MINI:ES1!` and `timeframe 5`). Output
// is time,open,high,low,close,volume with tz-naive UTC timestamps, the format
// quantlab.data.load_csv expects.

import fs from 'node:fs';
import { evaluate, disconnect, KNOWN_PATHS } from '../src/connection.js';

const CHART = KNOWN_PATHS.chartApi;
const BARS = KNOWN_PATHS.mainSeriesBars;

const out = process.argv[2] || 'bars.csv';
const untilTs = process.argv[3] ? Math.floor(new Date(process.argv[3]).getTime() / 1000) : 0;
const maxPages = Number(process.argv[4] || 600);

const iso = (t) => new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const info = await evaluate(`(function () {
  var c = ${CHART}; var s = null, r = null;
  try { s = c.symbol(); } catch (e) {}
  try { r = c.resolution(); } catch (e) {}
  return { symbol: s, resolution: r };
})()`);
process.stderr.write(`chart: ${info.symbol} @ ${info.resolution}\n`);

// ---- page back ----------------------------------------------------------
let last = null;
for (let page = 0; page < maxPages; page++) {
  const st = await evaluate(`(function () {
    var ms = ${CHART}._chartWidget.model().mainSeries();
    var b = ms.bars(); var fv = b.valueAt(b.firstIndex());
    var more = true; try { more = ms.requestMoreDataAvailable(); } catch (e) {}
    return { first: fv && fv[0], size: b.size(), more: more };
  })()`);
  if (page % 10 === 0 || !st.more) {
    process.stderr.write(`page ${page}: ${st.size} bars, earliest ${st.first ? iso(st.first) : '?'}, more=${st.more}\n`);
  }
  if (!st.more || (untilTs && st.first <= untilTs)) break;
  // a page that did not grow the series is the feed stalling, not the end;
  // give it one more wait before treating it as the end
  if (last !== null && st.size === last) {
    await sleep(3000);
    const again = await evaluate(`(function () { return ${BARS}.size(); })()`);
    if (again === last) { process.stderr.write(`no growth after page ${page}; stopping\n`); break; }
  }
  last = st.size;
  await evaluate(`(function () { try { ${CHART}._chartWidget.model().mainSeries().requestMoreData(1000); } catch (e) {} })()`);
  await sleep(1800);
}

// ---- read everything by index --------------------------------------------
const meta = await evaluate(`(function () { var b = ${BARS}; return { first: b.firstIndex(), last: b.lastIndex(), size: b.size() }; })()`);
const fd = fs.openSync(out, 'w');
fs.writeSync(fd, 'time,open,high,low,close,volume\n');
let n = 0, earliest = null, latest = null;
for (let s = meta.first; s <= meta.last; s += 5000) {
  const e = Math.min(meta.last, s + 4999);
  const rows = await evaluate(`(function () {
    var b = ${BARS}; var r = [];
    for (var i = ${s}; i <= ${e}; i++) { var v = b.valueAt(i); if (v && v[0]) r.push([v[0], v[1], v[2], v[3], v[4], v[5] || 0]); }
    return r;
  })()`);
  for (const v of rows) {
    fs.writeSync(fd, `${iso(v[0])},${v[1]},${v[2]},${v[3]},${v[4]},${v[5]}\n`);
    n++;
    if (earliest === null || v[0] < earliest) earliest = v[0];
    if (latest === null || v[0] > latest) latest = v[0];
  }
}
fs.closeSync(fd);
process.stderr.write(`wrote ${n} bars to ${out}: ${earliest ? iso(earliest) : '?'} -> ${latest ? iso(latest) : '?'}\n`);
await disconnect();
process.exit(0);
