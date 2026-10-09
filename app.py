"""app.py - mobile-first Streamlit UI for FX regime detection/forecast (DXY focus).

Analytical tool, not financial advice. Regimes are forecast; direction is shown only as descriptive context.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import core

# ----------------------------------------------------------------------------- constants
LWC_URL = "https://cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"
SHORT = ["Stagnant", "Steady", "Choppy", "Volatile"]


def _secret(name: str, default: str) -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


OWNER = _secret("OWNER", "")
REPO = _secret("REPO", "")
DATA_BRANCH = _secret("DATA_BRANCH", "data")
CFG = (OWNER, REPO, DATA_BRANCH)

st.set_page_config(page_title="FX Regime Monitor", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding:0.6rem 0.7rem 2rem 0.7rem}h1{font-size:1.35rem}"
            "div[data-testid='stMetricValue']{font-size:1.15rem}</style>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- cached loaders
@st.cache_data(ttl=300, show_spinner=False)
def load_intraday(interval: str, cfg: tuple):
    """Native intraday download (5 min cache)."""
    return core.load_native(interval, cfg)


@st.cache_data(ttl=3600, show_spinner=False)
def load_daily():
    """Native daily download (1 h cache)."""
    return core.load_native("1d", None)


def load(interval: str):
    return load_daily() if interval == "1d" else load_intraday(interval, CFG)


@st.cache_data(ttl=300, show_spinner=False)
def get_set(tf: str, anchor: int, choice: str, cfg: tuple, daily_stamp: int):
    """Closed-bar frames at a timeframe (cached; daily_stamp refreshes hourly for daily-based rungs)."""
    native, errs = load(core.TF_NATIVE[tf])
    return core.build_set(native, tf, anchor, choice), errs


def S_for(tf: str, anchor: int, choice: str):
    stamp = int(time.time() // 3600) if core.TF_NATIVE[tf] == "1d" else 0
    return get_set(tf, anchor if tf in ("2h", "3h", "4h") else 0, choice, CFG, stamp)


@st.cache_resource(show_spinner=False, max_entries=6)
def get_ml(symbol, tf, H, qa, qe, atr_n, anchor, choice, last_ts, cfg):
    """Walk-forward + live model; cached on (symbol, timeframe, H, quantiles, DXY source, last closed bar)."""
    S, _ = S_for(tf, anchor, choice)
    df = core.get_series(S, symbol)
    tgt = core.core_frame(df, atr_n, H)
    Xb = core.base_features(df, core.neighbour_frames(S, symbol), H, atr_n, tf)
    return core.run_ml(tgt, Xb, H, qa, qe)


# ----------------------------------------------------------------------------- small helpers
def hex_rgba(h: str, a: float) -> str:
    return f"rgba({int(h[1:3], 16)},{int(h[3:5], 16)},{int(h[5:7], 16)},{a})"


def chip(k: int) -> str:
    c = core.REGIME_COLORS[k]
    return (f"<span style='background:{hex_rgba(c, .25)};border:1px solid {c};padding:1px 8px;"
            f"border-radius:10px;font-weight:600'>{core.REGIME_NAMES[k]}</span>")


def html_table(headers, rows) -> None:
    th = "".join(f"<th style='text-align:left;padding:2px 6px;font-weight:600'>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td style='padding:2px 6px'>{c}</td>" for c in r) + "</tr>" for r in rows)
    st.markdown(f"<table style='font-size:13px;border-collapse:collapse;width:100%'><tr>{th}</tr>{body}</table>",
                unsafe_allow_html=True)


def prob_bars(p) -> str:
    out = ""
    for k in range(4):
        c = core.REGIME_COLORS[k]
        out += (f"<div style='display:flex;align-items:center;font-size:12px;margin:1px 0'>"
                f"<span style='width:96px'>{core.REGIME_NAMES[k]}</span>"
                f"<div style='flex:1;background:#8883;height:8px;border-radius:4px'>"
                f"<div style='width:{p[k] * 100:.0f}%;background:{c};height:8px;border-radius:4px'></div></div>"
                f"<span style='width:42px;text-align:right'>{p[k]:.0%}</span></div>")
    return out


def span_text(H: int, tf: str) -> str:
    h = H * core.TF_SECONDS[tf] / 3600.0
    return f"{h:g} hours" if h < 48 else f"{h / 24:g} days"


def disp_secs(idx: pd.DatetimeIndex, tz: str) -> np.ndarray:
    loc = idx.tz_convert(tz).tz_localize(None)
    return ((loc - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy().astype("int64")


def fmt_ts(ts: pd.Timestamp, tz: str) -> str:
    return ts.tz_convert(tz).strftime("%Y-%m-%d %H:%M") + f" {tz}"


def tz_options() -> list:
    try:
        import zoneinfo
        z = sorted(zoneinfo.available_timezones())
        if "UTC" in z:
            return z
    except Exception:
        pass
    return ["UTC", "Europe/London", "Europe/Stockholm", "America/New_York", "Asia/Tokyo"]


# ----------------------------------------------------------------------------- chart
def swing_points(h: np.ndarray, l: np.ndarray):
    """5-bar fractal swing highs/lows (confirmed 2 bars later)."""
    hi, lo = [], []
    for i in range(2, len(h) - 2):
        if h[i] > max(h[i - 2], h[i - 1], h[i + 1], h[i + 2]):
            hi.append(i)
        if l[i] < min(l[i - 2], l[i - 1], l[i + 1], l[i + 2]):
            lo.append(i)
    return hi, lo


def chart_html(cf, R, theta, gamma, tz, opts, levels, overlay, dark) -> str:
    """Three synchronized TradingView Lightweight Charts panes (v4 API)."""
    n0 = len(cf)
    sl = slice(max(0, n0 - 1500), n0)
    c = cf.iloc[sl]
    t = disp_secs(c.index, tz)
    keep = t > np.r_[-1, t[:-1]]
    c, t, Rs = c[keep], t[keep], R[sl][keep]
    e20 = core.ema(cf["close"], 20).iloc[sl][keep].to_numpy()
    e50 = core.ema(cf["close"], 50).iloc[sl][keep].to_numpy()

    def ser(vals):
        return [{"time": int(a), "value": float(v)} if np.isfinite(v) else {"time": int(a)} for a, v in zip(t, vals)]

    data = {
        "dark": dark, "intraday": bool(opts["intraday"]), "n": int(len(c)),
        "candles": [{"time": int(a), "open": float(o), "high": float(h), "low": float(l), "close": float(k)}
                    for a, o, h, l, k in zip(t, c["open"], c["high"], c["low"], c["close"])],
        "regime": [{"time": int(a), "value": 1,
                    "color": hex_rgba(core.REGIME_COLORS[r], 0.15) if r >= 0 else "rgba(0,0,0,0)"}
                   for a, r in zip(t, Rs)],
        "natr": ser(c["natr"].to_numpy()), "er": ser(c["er"].to_numpy()),
        "theta": float(theta), "gamma": float(gamma), "overlay": overlay, "levels": levels,
        "ema20": ser(e20) if opts["ema"] else [], "ema50": ser(e50) if opts["ema"] else [],
        "swh": [], "swl": [], "markers": [],
    }
    if opts["swing"]:
        hi, lo = swing_points(c["high"].to_numpy(), c["low"].to_numpy())
        data["swh"] = [{"time": int(t[i]), "value": float(c["high"].iloc[i])} for i in hi]
        data["swl"] = [{"time": int(t[i]), "value": float(c["low"].iloc[i])} for i in lo]
    if opts["markers"]:
        ch = np.where((Rs[1:] != Rs[:-1]) & (Rs[1:] >= 0) & (Rs[:-1] >= 0))[0] + 1
        data["markers"] = [{"time": int(t[i]), "position": "belowBar", "shape": "circle",
                            "color": core.REGIME_COLORS[Rs[i]], "text": ""} for i in ch]
    legend = "".join(f"<span style='margin-right:10px'><i style='display:inline-block;width:10px;height:10px;"
                     f"background:{hex_rgba(col, .6)};border:1px solid {col};margin-right:3px'></i>{nm}</span>"
                     for col, nm in zip(core.REGIME_COLORS, core.REGIME_NAMES))
    tpl = """
<div style="font-family:-apple-system,system-ui,sans-serif;width:100%">
<div id="c1" style="height:380px;position:relative;width:100%">
<div id="ov" style="position:absolute;top:6px;left:8px;z-index:5;font-size:11px;background:rgba(20,20,25,.65);color:#eee;padding:3px 6px;border-radius:4px;max-width:90%;pointer-events:none"></div></div>
<div id="c2" style="height:120px;width:100%"></div><div id="c3" style="height:120px;width:100%"></div>
<div style="font-size:11px;color:#888;padding:4px 2px;line-height:1.6">__LEGEND__<br>
pane 2: nATR (dashed = theta) | pane 3: ER (dashed = gamma)</div></div>
<script src="__URL__"></script>
<script>
(function(){
const D=__DATA__;
if(typeof LightweightCharts==='undefined'){document.getElementById('c1').innerHTML='<p style="padding:12px;color:#c33">Chart library failed to load (CDN blocked or offline).</p>';return;}
const bg=D.dark?'#0e1117':'#ffffff',tc=D.dark?'#cbd5e1':'#334155',gc=D.dark?'#1f2937':'#e5e7eb';
function mk(id,h){const el=document.getElementById(id);return LightweightCharts.createChart(el,{width:el.clientWidth,height:h,
layout:{background:{type:'solid',color:bg},textColor:tc,fontSize:11},grid:{vertLines:{color:gc},horzLines:{color:gc}},
rightPriceScale:{borderColor:gc},timeScale:{borderColor:gc,timeVisible:D.intraday,secondsVisible:false,rightOffset:3},
handleScroll:{vertTouchDrag:false}});}
const c1=mk('c1',380),c2=mk('c2',120),c3=mk('c3',120);
const rg=c1.addHistogramSeries({priceScaleId:'regime',priceLineVisible:false,lastValueVisible:false,
autoscaleInfoProvider:()=>({priceRange:{minValue:0,maxValue:1}})});
c1.priceScale('regime').applyOptions({scaleMargins:{top:0,bottom:0},visible:false});
rg.setData(D.regime);
const cs=c1.addCandlestickSeries({upColor:'#26a69a',downColor:'#ef5350',borderVisible:false,wickUpColor:'#26a69a',wickDownColor:'#ef5350'});
cs.setData(D.candles);
function ln(ch,color,data,style){const s=ch.addLineSeries({color:color,lineWidth:1,priceLineVisible:false,lastValueVisible:false,lineStyle:style||0});s.setData(data);return s;}
if(D.ema20.length){ln(c1,'#facc15',D.ema20);ln(c1,'#a78bfa',D.ema50);}
if(D.swh.length){ln(c1,'#94a3b8',D.swh,2);}
if(D.swl.length){ln(c1,'#94a3b8',D.swl,2);}
if(D.markers.length){cs.setMarkers(D.markers);}
D.levels.forEach(l=>cs.createPriceLine({price:l.p,color:l.c,lineWidth:1,lineStyle:2,axisLabelVisible:true,title:l.t}));
const s2=ln(c2,'#38bdf8',D.natr);
if(isFinite(D.theta))s2.createPriceLine({price:D.theta,color:'#f59e0b',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:'theta'});
const s3=ln(c3,'#34d399',D.er);
if(isFinite(D.gamma))s3.createPriceLine({price:D.gamma,color:'#f59e0b',lineWidth:1,lineStyle:2,axisLabelVisible:true,title:'gamma'});
document.getElementById('ov').innerHTML=D.overlay;
const cs_=[c1,c2,c3];let lock=false;
cs_.forEach((ch,i)=>ch.timeScale().subscribeVisibleLogicalRangeChange(r=>{if(lock||!r)return;lock=true;
cs_.forEach((o,j)=>{if(j!==i)o.timeScale().setVisibleLogicalRange(r);});lock=false;}));
c1.timeScale().setVisibleLogicalRange({from:Math.max(0,D.n-150),to:D.n+3});
window.addEventListener('resize',()=>{cs_.forEach((ch,i)=>ch.applyOptions({width:document.getElementById('c'+(i+1)).clientWidth}));});
})();
</script>"""
    return (tpl.replace("__DATA__", json.dumps(data)).replace("__URL__", LWC_URL).replace("__LEGEND__", legend))


# ----------------------------------------------------------------------------- settings
st.title("FX Regime Monitor")
with st.expander("Settings", expanded=False):
    c1, c2 = st.columns(2)
    symbol = c1.selectbox("Instrument", core.TARGETS, index=0)
    choice = c2.radio("DXY source", ["Auto", "Real", "Synthetic"], horizontal=True)
    c1, c2 = st.columns(2)
    tf = c1.selectbox("Timeframe", core.TF_ORDER, index=1)
    pres = core.PRESETS[tf]
    labels = [f"{h} bars ({lab})" for h, lab, _ in pres] + ["Custom"]
    dflt = [i for i, p in enumerate(pres) if p[2]][0]
    pick = c2.selectbox("Horizon", labels, index=dflt, key=f"hz_{tf}")
    if pick == "Custom":
        H = int(st.number_input("Custom horizon (bars)", 3, 300, core.default_H(tf), key=f"hc_{tf}"))
    else:
        H = pres[labels.index(pick)][0]
    H = max(3, H)
    c1, c2 = st.columns(2)
    qa = c1.slider("ATR percentile (theta)", 30, 70, 50)
    qe = c2.slider("ER percentile (gamma)", 30, 70, 50)
    c1, c2 = st.columns(2)
    atr_n = int(c1.number_input("ATR length N", 3, 50, 10))
    tz = c2.selectbox("Timezone", tz_options(), index=tz_options().index("UTC") if "UTC" in tz_options() else 0)
    anchor = 0
    if tf in ("2h", "3h", "4h"):
        b = int(tf[0])
        anchor = st.slider("UTC anchor offset (hours)", 0, b - 1, 0)
    t1, t2, t3, t4, t5 = st.columns(5)
    o_ema = t1.toggle("EMA", True)
    o_sw = t2.toggle("Swings", False)
    o_mk = t3.toggle("Changes", True)
    o_lv = t4.toggle("PDH/PWH", True)
    o_dark = t5.toggle("Dark", True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

# ----------------------------------------------------------------------------- data
with st.spinner("Loading data..."):
    S0, errs = S_for(tf, anchor, choice)
    resolved = "Real" if S0["use_real"] else "Synthetic"
    S, _ = S_for(tf, anchor, resolved)
df = core.get_series(S, symbol)
FOOT = "Analytical tool, not financial advice."
warns = list(errs) + list(S["warnings"])
if len(df) < 30:
    st.error("No usable data for this selection (Yahoo returned too little data). Try Refresh data, "
             "another timeframe, or retry later.")
    for w in warns:
        st.warning(w)
    st.caption(FOOT)
    st.stop()

now = pd.Timestamp.now(tz="UTC")
last_ts = df.index[-1]
cf = core.core_frame(df, atr_n, H)
with st.spinner("Training CPU regime model (about 20 s on first run)..."):
    ml = get_ml(symbol, tf, H, qa, qe, atr_n, anchor, resolved, str(last_ts), CFG)
if ml["ok"]:
    theta, gamma = ml["theta"], ml["gamma"]
else:
    theta, gamma = core.fit_thresholds(cf["natr"], cf["er"], qa, qe)
R = core.regimes(cf["natr"], cf["er"], theta, gamma)
ylab = core.make_labels(R, H)
age_arr = core.run_length(R)
cur_R = int(R[-1])
age = int(age_arr[-1])
live = ml.get("live") if ml["ok"] else None
bias = core.bias_detail(cf, gamma, H)
wend = core.window_end(last_ts, tf, H)
stale = (now.weekday() < 5 and tf in ("30m", "1h", "2h", "3h", "4h", "1D")
         and (now - last_ts).total_seconds() > (4 * 86400 if tf == "1D" else 3 * core.TF_SECONDS[tf] + 1800))
if stale:
    warns.append("Data may be stale: the last closed bar is older than expected for a weekday.")
if tf == "30m":
    warns.append("30m: only ~60 days of history are available from Yahoo.")
if ml["ok"]:
    warns += ml["warnings"]
elif ml.get("msg"):
    warns.append(ml["msg"])

# multi-timeframe table (descriptive)
rungs = [tf] + [r for r in core.LADDER if core.TF_ORDER.index(r) > core.TF_ORDER.index(tf)]
mtf = []
with st.spinner("Building multi-timeframe context..."):
    for r in rungs:
        if r == tf:
            row = core.mtf_row(df, H, qa, qe, atr_n)
        else:
            Sr, _ = S_for(r, anchor if r in ("2h", "3h", "4h") else 0, resolved)
            row = core.mtf_row(core.get_series(Sr, symbol), core.default_H(r), qa, qe, atr_n)
        mtf.append((r, row))
valid_m = [m for m in mtf if m[1]]
up = sum(1 for _, m in valid_m if m["bias"] == 1)
dn = sum(1 for _, m in valid_m if m["bias"] == -1)
if valid_m and max(up, dn) > 0 and up != dn:
    align = f"{max(up, dn)} of {len(valid_m)} timeframes: {'up' if up > dn else 'down'} bias"
elif valid_m:
    align = f"Mixed: {up} up, {dn} down, {len(valid_m) - up - dn} neutral of {len(valid_m)} timeframes"
else:
    align = "Multi-timeframe alignment unavailable (insufficient history)"

# prior day/week levels
levels = []
if o_lv:
    for r, lab_h, lab_l, colr in (("1D", "PDH", "PDL", "#60a5fa"), ("1W", "PWH", "PWL", "#c084fc")):
        try:
            Sr, _ = S_for(r, 0, resolved)
            dd = core.get_series(Sr, symbol)
            if len(dd):
                levels += [{"p": float(dd["high"].iloc[-1]), "c": colr, "t": lab_h},
                           {"p": float(dd["low"].iloc[-1]), "c": colr, "t": lab_l}]
        except Exception:
            pass

pip = core.pip_size(symbol)
atr_pips = cf["atr"].iloc[-1] / pip if np.isfinite(cf["atr"].iloc[-1]) else np.nan
nat = cf["natr"].dropna()
nat_pct = float((nat <= nat.iloc[-1]).mean()) if len(nat) else np.nan
unit = "points" if symbol == "DXY" else "pips"
prov = (f"Yahoo (free, unofficial) | last closed bar {fmt_ts(last_ts, 'UTC')} | {len(df)} bars | "
        f"{df.index[0].strftime('%Y-%m-%d')} to {last_ts.strftime('%Y-%m-%d')} | "
        f"{'derived from ' + core.TF_NATIVE[tf] + ' bars | ' if tf not in ('30m', '1h', '1D') else ''}"
        f"{S['source_label'] if symbol == 'DXY' else 'DXY neighbour: ' + S['source_label']}"
        f"{' | synthetic series in use' if (symbol == 'DXY' and not S['use_real']) else ''}")


def head() -> None:
    st.caption(prov)
    if symbol == "DXY":
        st.markdown(f"**{S['source_label']}**")


T_chart, T_fc, T_dir, T_ctx, T_stats, T_method = st.tabs(
    ["Chart", "Forecast", "Direction", "Context", "Stats", "Method"])

# ----------------------------------------------------------------------------- Chart tab
with T_chart:
    head()
    for w in warns:
        st.warning(w)
    if age <= 3 and age < len(R) and R[-1 - age] >= 0:
        st.info(f"Regime changed within the last 3 bars (now {core.REGIME_NAMES[cur_R]}).")
    if live is not None:
        top = np.argsort(live)[::-1]
        fc_txt = (f"Model estimate for the next {H} bars (about {span_text(H, tf)}): " +
                  ", ".join(f"{core.REGIME_NAMES[k]} {live[k]:.0%}" for k in top[:3]) + ".")
    else:
        fc_txt = "No model forecast is available for this selection (rule-based layer only)."
    if ml["ok"]:
        mb, mn = ml["metrics"][ml["best"]]["bal"], ml["metrics"]["Naive"]["bal"]
        vs = (f"Out-of-sample balanced accuracy: model {mb:.2f} vs naive persistence {mn:.2f} "
              f"({'model beat naive' if ml['beats_naive'] else 'model did NOT beat naive'}).")
    else:
        vs = "No out-of-sample test was possible."
    bias_txt = (f"Direction context (past bars, not a forecast): {bias.get('label', 'n/a')}; {align}." if bias else "")
    st.info(f"{symbol} {tf} is in the **{core.REGIME_NAMES[cur_R]}** regime (age {age} bars). {fc_txt} {bias_txt} "
            f"ATR is {atr_pips:.{2 if symbol == 'DXY' else 1}f} {unit}. {vs}")
    c1, c2 = st.columns(2)
    c1.markdown(f"**Current regime**<br>{chip(cur_R)} age {age} bars", unsafe_allow_html=True)
    if live is not None:
        k = int(np.argmax(live))
        c2.markdown(f"**Next {H} bars** (model estimate)<br>{chip(k)} {live[k]:.0%}", unsafe_allow_html=True)
        c2.markdown(prob_bars(live), unsafe_allow_html=True)
    else:
        c2.markdown("**Next bars**<br>forecast unavailable", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.markdown(f"**Direction bias**<br>{bias.get('label', 'n/a')} <span style='font-size:12px'>"
                f"(describes past bars)</span>", unsafe_allow_html=True)
    c2.markdown(f"**Multi-timeframe**<br>{align}", unsafe_allow_html=True)
    if live is not None:
        ov_fc = " ".join(f"<span style='color:{core.REGIME_COLORS[k]}'>{core.REGIME_NAMES[k]} {live[k]:.0%}</span>"
                         for k in np.argsort(live)[::-1][:3])
    else:
        ov_fc = "n/a"
    overlay = (f"Now: <b style='color:{core.REGIME_COLORS[cur_R]}'>{core.REGIME_NAMES[cur_R]}</b> (age {age} bars)"
               f" | Next {H} bars: {ov_fc}")
    components.html(chart_html(cf, R, theta, gamma, tz, {"ema": o_ema, "swing": o_sw, "markers": o_mk,
                                                        "intraday": tf not in ("1D", "3D", "1W")},
                               levels, overlay, o_dark), height=700, scrolling=False)
    st.caption("Regime colours use thresholds fitted on all labeled rows (descriptive). Probabilities are model "
               "estimates, not facts. The forming bar is excluded.")

# ----------------------------------------------------------------------------- Forecast tab
with T_fc:
    head()
    st.markdown(f"**Window:** next {H} bars (about {span_text(H, tf)}), until about {fmt_ts(wend, tz)}; "
                "weekend closures can extend the clock time.")
    st.caption("Forecast = probability of the regime TYPE over a rolling window after the last closed bar; not a "
               "direction forecast. For the rest of today use the 6h or 12h presets.")
    st.markdown("**CPU regime model (graph-inspired features)**")
    if not ml["ok"]:
        st.warning(ml.get("msg", "Model unavailable."))
        st.markdown(f"Rule-based regime now: {chip(cur_R)}", unsafe_allow_html=True)
    else:
        if live is not None:
            st.markdown(prob_bars(live), unsafe_allow_html=True)
            st.caption(f"Model estimate ({ml['best']}, chosen by walk-forward balanced accuracy), not a fact.")
        met = ml["metrics"]
        rows = [[k, f"{m['acc']:.3f}", f"{m['bal']:.3f}", f"{m['f1']:.3f}"] for k, m in met.items()]
        st.dataframe(pd.DataFrame(rows, columns=["Model", "Acc", "BalAcc", "F1"]), hide_index=True, width="stretch")
        if not ml["beats_naive"]:
            st.warning("The model did NOT beat naive persistence on balanced accuracy out of sample. "
                       "Treat the probabilities with caution.")
        else:
            st.success("The model beat naive persistence on balanced accuracy out of sample (a historical "
                       "test, not a guarantee).")
        rc = pd.DataFrame({k: [f"{x:.2f}" if np.isfinite(x) else "-" for x in m["recall"]] for k, m in met.items()},
                          index=SHORT)
        rc["Support"] = met["HGB"]["support"]
        st.markdown("**Per-class recall and support**")
        st.dataframe(rc.reset_index().rename(columns={"index": "Class"}), hide_index=True, width="stretch")
        b = ml["best"]
        for nm in (b, "Naive"):
            st.markdown(f"**Confusion matrix: {nm}** (rows true, cols predicted)")
            st.dataframe(pd.DataFrame(met[nm]["cm"], index=SHORT, columns=[s[:3] for s in SHORT])
                         .reset_index().rename(columns={"index": "True"}), hide_index=True, width="stretch")
        with st.expander("Trading risk score"):
            st.caption("R = sum(CM * C) / N with the paper's example cost matrix (rows true, cols predicted). "
                       "Example, trend-following perspective. Lower is better.")
            st.dataframe(pd.DataFrame([[k, f"{m['risk']:.3f}"] for k, m in met.items()], columns=["Model", "Risk"]),
                         hide_index=True, width="stretch")
        st.markdown("**Outcome separation** (out-of-sample, grouped by predicted class)")
        sp = ml["sep"].copy()
        sp.insert(0, "Class", SHORT)
        sp = sp.rename(columns={"n": "N", "move": "|move|/ATR", "rng": "range/ATR", "trend": "Trend later"})
        sp["Trend later"] = sp["Trend later"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "-")
        for cn in ("|move|/ATR", "range/ATR"):
            sp[cn] = sp[cn].map(lambda v: f"{v:.2f}" if np.isfinite(v) else "-")
        sp["N"] = sp["N"].fillna(0).astype(int)
        st.dataframe(sp, hide_index=True, width="stretch")
        st.markdown("**Calibration** (top-class probability vs observed accuracy)")
        cal = ml["cal"].copy()
        cal = pd.DataFrame({"Bin": cal.index.astype(str), "N": cal["n"].fillna(0).astype(int),
                            "Observed": cal["acc"].map(lambda v: f"{v:.0%}" if np.isfinite(v) else "-")})
        st.dataframe(cal, hide_index=True, width="stretch")
        if ml["imp"] is not None:
            st.markdown("**Top-10 features** (" + ("permutation importance, last test block"
                                                   if ml["best"] == "HGB" else "mean |LR coefficient|") + ")")
            st.dataframe(ml["imp"].rename("Importance").round(4).reset_index().rename(columns={"index": "Feature"}),
                         hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- Direction tab
with T_dir:
    head()
    st.caption("Descriptive context from past bars; not a forecast and not a signal.")
    if bias:
        st.markdown(f"**{bias['label']}** | signed ER {bias['ser']:+.2f} | net move {bias['net_atr']:+.1f} ATR over "
                    f"{H} bars | {bias['ema_state']} | EMA20 slope {bias['slope_atr']:+.2f} ATR/5 bars")
        st.caption("Neutral if |signed ER| < gamma or the EMA state disagrees.")
    st.markdown("**Direction persistence evidence**")
    raw_sign = np.sign(cf["ser"].fillna(0).to_numpy())
    pe = core.persistence(cf["close"].to_numpy(), raw_sign, R, H)
    html_table(["Regime group", "Hit rate", "N", "95% CI"],
               [[p["group"], f"{p['rate']:.0%}" if p["n"] else "-", p["n"],
                 f"{p['lo']:.0%}-{p['hi']:.0%}" if p["n"] else "-"] for p in pe])
    st.caption(f"Share of times sign(C[T+{H}] - C[T]) matched the sign of signed ER at T; non-overlapping samples "
               "(every H-th bar), Wilson interval. Rates near 50% mean no persistence evidence.")
    for p in pe:
        if p["n"] and abs(p["rate"] - 0.5) < 0.05:
            st.info(f"{p['group']}: hit rate is close to 50%, i.e. direction at T says little about direction later.")
    st.markdown("**Multi-timeframe (descriptive)**")
    rows = []
    for r, m in mtf:
        if m is None:
            rows.append([r, "insufficient history", "-", "-", "-"])
        else:
            rows.append([r, chip(m["regime"]), {1: "Up", -1: "Down", 0: "Neutral"}[m["bias"]],
                         f"{m['er']:.2f}", f"{m['pct']:.0%}"])
    html_table(["TF", "Regime", "Bias", "ER", "nATR pct"], rows)
    st.markdown(f"**{align}**")
    st.caption("Thresholds for each timeframe come from its full sample (descriptive, not out-of-sample).")

# ----------------------------------------------------------------------------- Context tab
with T_ctx:
    head()
    st.markdown("**Volatility context**")
    st.markdown(f"ATR({atr_n}) = **{atr_pips:.{2 if symbol == 'DXY' else 1}f} {unit}** | nATR percentile {nat_pct:.0%}")
    cs_df = core.conditional_stats(cf, R, H)
    html_table(["R(T)", f"|move|/ATR med / p80", "H-range/ATR med / p80", "N"],
               [[chip(int(r.regime)), f"{r.mv_med:.2f} / {r.mv_p80:.2f}" if r.n else "-",
                 f"{r.rg_med:.2f} / {r.rg_p80:.2f}" if r.n else "-", int(r.n)] for r in cs_df.itertuples()])
    st.caption(f"Empirical history over horizon H={H} (overlapping windows; not a forecast). "
               f"Current regime highlighted: {core.REGIME_NAMES[cur_R]}.")
    if tf in ("30m", "1h", "2h", "3h", "4h"):
        st.markdown("**Session profile (UTC)**")
        byh, byd = core.session_profile(cf["natr"])
        if len(byh):
            cur_h = last_ts.hour
            hdf = pd.DataFrame({"other": byh.where(byh.index != cur_h), "current hour": byh.where(byh.index == cur_h)})
            st.bar_chart(hdf, color=["#64748b", "#f59e0b"], height=150)
            ddf = pd.DataFrame({"nATR": byd.rename(index={i: n for i, n in enumerate(
                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])})})
            st.bar_chart(ddf, height=130)
    if symbol == "DXY":
        st.markdown("**DXY basket**")
        cdf, tot = core.contributions(S["pairs"], H)
        if cdf is None:
            st.info("Not enough aligned history for the basket decomposition.")
        else:
            cdf = cdf.reindex(cdf["contrib"].abs().sort_values(ascending=False).index)
            mx = max(cdf["contrib"].abs().max(), 1e-12)
            rows = []
            for r in cdf.itertuples():
                pr = core.mtf_row(S["pairs"][r.pair], H, qa, qe, atr_n)
                bar = "█" * int(round(8 * abs(r.contrib) / mx))
                rows.append([r.pair, f"{r.weight:.1%}", chip(pr["regime"]) if pr else "-",
                             {1: "USD up", -1: "USD down", 0: "flat"}[int(r.usd_dir)],
                             f"{r.contrib * 100:+.3f}% {bar}"])
            html_table(["Pair", "Wt", "Regime", "USD", f"Contrib ({H} bars)"], rows)
            st.caption("contrib = exponent x ln(P_t / P_t-H); positive pushed DXY up; contributions sum exactly to "
                       f"ln(DXY_t / DXY_t-H) of the synthetic series ({tot * 100:+.3f}%).")
            if S["use_real"] and len(S["real"]) > H:
                rl = float(np.log(S["real"]["close"].iloc[-1] / S["real"]["close"].iloc[-1 - H]))
                st.markdown(f"Real DXY log change {rl * 100:+.3f}% | unexplained (real minus contributions): "
                            f"{(rl - tot) * 100:+.3f}%")
            signs = cdf["usd_dir"].to_numpy()
            breadth = float((cdf["weight"].to_numpy() * signs).sum())
            trend_share = np.mean([1 if (core.mtf_row(S["pairs"][p], H, qa, qe, atr_n) or {"regime": 0})["regime"]
                                   in (1, 3) else 0 for p in core.PAIRS])
            st.markdown(f"Weighted USD breadth: **{breadth:+.2f}** (-1 all USD down, +1 all USD up) | "
                        f"basket trend breadth: **{trend_share:.0%}** of pairs in a trending regime")
        v = S["validation"]
        st.markdown("**Validation (synthetic vs real DX-Y.NYB)**")
        if v:
            st.markdown(f"Return correlation {v['corr']:.3f} | mean abs level difference {v['mad']:.3f} | "
                        f"overlapping bars {v['n']}")
        else:
            st.caption("Real DX-Y.NYB not available for this selection; validation not possible.")
    st.markdown("**Cross-asset**")
    eu = S["pairs"]["EURUSD"]
    ref = core.get_series(S, "DXY") if symbol != "DXY" else df
    j = pd.concat([eu["close"].pct_change().rename("e"), ref["close"].pct_change().rename("d")],
                  axis=1, join="inner").dropna()
    if symbol != "EURUSD":
        if len(j) >= 100:
            cor = j["e"].rolling(100).corr(j["d"]).iloc[-1]
            r_eu = core.mtf_row(eu, H, qa, qe, atr_n)
            r_dx = core.mtf_row(ref, H, qa, qe, atr_n)
            agree = ""
            if r_eu and r_dx:
                same = r_eu["regime"] == r_dx["regime"]
                typ = (r_eu["regime"] in (1, 3)) == (r_dx["regime"] in (1, 3))
                agree = f" | regime agreement: {'same regime' if same else ('same type' if typ else 'disagree')}"
            st.markdown(f"Rolling 100-bar return correlation EURUSD vs DXY: **{cor:+.2f}**{agree}")
        else:
            st.caption("Not enough overlapping bars for the correlation.")
    else:
        st.caption("Select DXY or another pair for the EURUSD vs DXY comparison.")
    st.markdown("**Regime meanings**")
    for m in core.MEANINGS:
        st.markdown(f"- {m}")

# ----------------------------------------------------------------------------- Stats tab
with T_stats:
    head()
    st.markdown("**Regime dynamics**")
    st.markdown(f"Current: {chip(cur_R)} age {age} bars", unsafe_allow_html=True)
    tm = core.transition_matrix(R, ylab)
    dur = core.median_durations(R)
    tdf = pd.DataFrame([[f"{v:.0%}" if np.isfinite(v) else "-" for v in row] for row in tm],
                       index=SHORT, columns=[s[:3] for s in SHORT])
    tdf["Med dur"] = [f"{d:g}" if np.isfinite(d) else "-" for d in dur]
    st.dataframe(tdf.reset_index().rename(columns={"index": "R(T) to next-H"}), hide_index=True, width="stretch")
    st.caption(f"Empirical transitions from R(T) to the label over the next {H} bars; durations in bars.")
    exp = pd.DataFrame({"time_utc": cf.index, **{c: cf[c].to_numpy() for c in core.OHLC},
                        "atr": cf["atr"].to_numpy(), "natr": cf["natr"].to_numpy(), "er": cf["er"].to_numpy(),
                        "regime": R, "label": ylab}).set_index("time_utc")
    if ml["ok"]:
        oos = ml["oos"]
        pcol = "h" if ml["best"] == "HGB" else "l"
        for k in range(4):
            exp[f"p{k}_oos"] = np.nan
            exp.iloc[oos["pos"].to_numpy(), exp.columns.get_loc(f"p{k}_oos")] = oos[f"{pcol}{k}"].to_numpy()
        if live is not None:
            for k in range(4):
                exp.iloc[-1, exp.columns.get_loc(f"p{k}_oos")] = live[k]
    Xb = core.base_features(df, core.neighbour_frames(S, symbol), H, atr_n, tf)
    exp = exp.join(Xb.add_prefix("f_").drop(columns=["f_natr", "f_er"], errors="ignore")).tail(5000)
    st.download_button("Download CSV (regimes, probabilities, features)", exp.to_csv().encode(),
                       file_name=f"{symbol}_{tf}_regimes.csv", mime="text/csv")
    with st.expander("Diagnostics"):
        st.markdown(f"Rows: target {len(df)} | labeled {ml.get('n_labeled', 'n/a')} | "
                    f"first bar {df.index[0]} | last bar {last_ts}")
        st.markdown(f"Model training time: {ml['train_time']:.1f} s" if ml["ok"] else "Model not trained.")
        st.markdown(f"theta {theta:.6f} | gamma {gamma:.3f} | DXY choice: {choice} -> {S['source_reason']}")
        if ml["ok"]:
            st.dataframe(pd.DataFrame(ml["folds"]).round(5), hide_index=True, width="stretch")
        for p in core.PAIRS:
            st.caption(f"{p}: {len(S['pairs'][p])} bars")
        for w in warns:
            st.caption(f"Warning: {w}")

# ----------------------------------------------------------------------------- Method tab
with T_method:
    head()
    st.markdown("""
**What this is.** A CPU-friendly *adaptation* inspired by ABN AMRO's 2005 FX Regime Prediction Indicator
(trend-vs-range probability from volatility and positioning) and the 2026 paper *Macroeconomic Message Passing for
Anticipating Foreign Exchange Regime Changes: A Deep Logical Learning Approach using Graph Tsetlin Machines*
(arXiv 2607.06719). Those need a GPU or proprietary data. This app is **not a replication**.

**Model.** The model here is called *CPU regime model (graph-inspired features)*: gradient boosting and logistic
regression on own features plus neighbour-pair features (the "graph" idea). The real Graph Tsetlin Machine requires an
NVIDIA GPU (pycuda/CUDA) and is **not run in this app**.

**Regimes.** True range, Wilder ATR (N adjustable), nATR = ATR/close, Efficiency Ratio over H bars.
theta and gamma are percentiles of nATR and ER fitted on training rows only. 0 Stagnant, 1 Steady trend, 2 Choppy,
3 Volatile trend. Label for a bar T = most frequent regime over the next H bars (ties go to the last bar).

**Forecast.** The probability of the regime *type* over the next H bars after the last closed bar. It is not a
direction forecast. Direction items are descriptive statistics of past bars.

**Validation and leakage.** Walk-forward, expanding window, first 50% minimum training set, remaining 50% in 4 test
blocks. Training rows whose label window reaches the test block are purged. Thresholds, labels, scalers and
percentile ranks are fitted on training data only; features use data up to bar T. Baselines: naive persistence and
majority class. Probabilities are model estimates and can be miscalibrated (see the calibration table).

**Data.** Yahoo Finance via yfinance (free, unofficial, may be gappy). 30m has about 60 days, 1h about 730 days.
2h/3h/4h are resampled from 1h, 3D from daily bars, 1W from daily bars (weeks ending Friday). The forming bar is
excluded. Yahoo FX has no volume. Synthetic DXY = 50.14348112 x EURUSD^-0.576 x USDJPY^0.136 x GBPUSD^-0.119 x
USDCAD^0.091 x USDSEK^0.042 x USDCHF^0.036; its high/low are an upper bound of the true range.

**Not provided.** No buy/sell signals, no price-direction prediction, no stop/target or position sizing.
""")
st.divider()
st.caption(FOOT)
