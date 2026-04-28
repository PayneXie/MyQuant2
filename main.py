from __future__ import annotations

import asyncio
import datetime as dt
import time
from typing import Iterable

import akshare as ak
import pandas as pd
import requests
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register


ETF_SYMBOL = "588810"
INDEX_SYMBOLS = ["sh000001", "sz399001"]

WINDOW_SIZE = 30
VOL_MULTIPLIER = 1.0

AVG_INCLUDES_TODAY = False

PRICE_CONFIRM_MODE = "none"
MA_WINDOW = 20


def _pick(df: pd.DataFrame, cols: list[str]) -> str | None:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _disable_akshare_tqdm() -> None:
    try:
        import importlib

        import akshare.utils.tqdm as ak_tqdm

        def _get_tqdm_noop(*args, **kwargs):
            return lambda it=None, **kw: it

        ak_tqdm.get_tqdm = _get_tqdm_noop
        for mod_name in [
            "akshare.utils.func",
            "akshare.index.index_stock_zh",
        ]:
            try:
                m = importlib.import_module(mod_name)
                if hasattr(m, "get_tqdm"):
                    setattr(m, "get_tqdm", _get_tqdm_noop)
            except Exception:
                continue
    except Exception:
        return


def _fetch_index_spot_amount_sina(symbols: list[str]) -> dict[str, float]:
    import re

    last_err = None
    for i in range(3):
        try:
            sym_list = ",".join([str(s).strip() for s in symbols])
            url = f"https://hq.sinajs.cn/list={sym_list}"
            headers = {"Referer": "https://finance.sina.com.cn", "User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers, timeout=10)
            res.encoding = "gbk"
            text = res.text
            out: dict[str, float] = {str(s).strip(): 0.0 for s in symbols}
            for sym, body in re.findall(r'var\s+hq_str_([^=]+)="([^"]*)";', text, flags=re.S):
                sym = str(sym).strip()
                parts = body.split(",")
                if len(parts) <= 9:
                    continue
                v = pd.to_numeric(parts[9], errors="coerce")
                out[sym] = float(v) if pd.notna(v) else 0.0
            return out
        except Exception as ex:
            last_err = ex
            time.sleep(0.4 * (i + 1))
    raise last_err if last_err is not None else RuntimeError("fetch index spot sina failed")


def _get_recent_trade_dates(end_dt: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    df = ak.tool_trade_date_hist_sina()
    df.columns = [str(c).strip() for c in df.columns]
    if "trade_date" not in df.columns:
        raise ValueError("获取交易日历失败: 缺少 trade_date 列")
    trade_dates = pd.to_datetime(df["trade_date"], errors="coerce")
    trade_dates = trade_dates.dropna().sort_values()
    trade_dates = trade_dates[trade_dates <= end_dt]
    return [pd.Timestamp(d).normalize() for d in trade_dates.tail(int(n)).tolist()]


def _fetch_sse_stock_amount_yuan(date_yyyymmdd: str) -> float:
    df = ak.stock_sse_deal_daily(date=date_yyyymmdd)
    df.columns = [str(c).strip() for c in df.columns]
    col_item = _pick(df, ["单日情况", "项目"])
    col_stock = _pick(df, ["股票"])
    if col_item is None or col_stock is None:
        raise ValueError("上交所成交概况缺少必要列")
    rows = df[df[col_item].astype(str).str.strip() == "成交金额"]
    if rows.empty:
        raise ValueError("上交所成交概况缺少 成交金额 行")
    v = pd.to_numeric(rows.iloc[0][col_stock], errors="coerce")
    if pd.isna(v):
        raise ValueError("上交所成交金额无法解析")
    return float(v) * 1e8


def _fetch_szse_stock_amount_yuan(date_yyyymmdd: str) -> float:
    df = ak.stock_szse_summary(date=date_yyyymmdd)
    df.columns = [str(c).strip() for c in df.columns]
    col_cat = _pick(df, ["证券类别"])
    col_amount = _pick(df, ["成交金额", "成交金额(元)", "成交额"])
    if col_cat is None or col_amount is None:
        raise ValueError("深交所总貌缺少必要列")
    rows = df[df[col_cat].astype(str).str.strip() == "股票"]
    if rows.empty:
        raise ValueError("深交所总貌缺少 股票 行")
    v = pd.to_numeric(rows.iloc[0][col_amount], errors="coerce")
    if pd.isna(v):
        raise ValueError("深交所成交金额无法解析")
    v = float(v)
    if v < 1e7:
        v = v * 1e8
    return v


def _fetch_total_money_series_exch(end_dt: pd.Timestamp, n_days: int) -> pd.Series:
    dates = _get_recent_trade_dates(end_dt=end_dt, n=n_days)
    values = []
    for d in dates:
        ymd = d.strftime("%Y%m%d")
        total = _fetch_sse_stock_amount_yuan(ymd) + _fetch_szse_stock_amount_yuan(ymd)
        values.append(total)
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name="total_money")


def _fetch_etf_daily_close(symbol: str, start_ymd: str, end_ymd: str) -> pd.DataFrame:
    import re

    raw = str(symbol).strip()
    s = raw.lower()
    variants: list[str] = []
    if re.fullmatch(r"(sh|sz)\d{6}", s):
        variants = [s, s[2:]]
    elif re.fullmatch(r"\d{6}", s):
        variants = [s, f"sh{s}", f"sz{s}"]
    else:
        variants = [raw]

    last_err: Exception | None = None
    for sym in variants:
        try:
            df = ak.fund_etf_hist_sina(symbol=sym)
        except Exception as ex:
            last_err = ex
            continue
        if df is None or getattr(df, "empty", True):
            continue
        df.columns = [str(c).strip() for c in df.columns]
        col_date = _pick(df, ["日期", "date", "交易日期"])
        col_close = _pick(df, ["收盘", "close"])
        if col_date is None or col_close is None:
            last_err = ValueError("ETF日线数据缺少必要列")
            continue
        out = pd.DataFrame({"date": pd.to_datetime(df[col_date]), "close": pd.to_numeric(df[col_close], errors="coerce")})
        out = out.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
        start_dt = pd.to_datetime(start_ymd)
        end_dt = pd.to_datetime(end_ymd)
        out = out[(out["date"] >= start_dt) & (out["date"] <= end_dt)].reset_index(drop=True)
        if out.empty:
            continue
        return out

    if last_err is not None:
        raise last_err
    raise ValueError("ETF日线数据为空")


def _calc_ma_price(
    etf_daily_close: pd.DataFrame,
    asof_dt: pd.Timestamp,
    window: int,
) -> tuple[float | None, pd.Timestamp | None, pd.Timestamp | None, pd.Timestamp | None]:
    df = etf_daily_close.copy()
    if df.empty:
        return None, None, None, None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return None, None, None, None

    dates = pd.DatetimeIndex(df["date"]).normalize()
    df = df.assign(date=dates)
    df = df[df["date"] <= asof_dt.normalize()].reset_index(drop=True)
    if df.empty:
        return None, None, None, None

    if int(window) <= 0:
        return None, None, None, None
    tail = df.tail(int(window))
    if tail.shape[0] < int(window):
        return None, None, None, None

    ma = float(tail["close"].mean())
    start_dt = pd.Timestamp(tail["date"].iloc[0]).normalize()
    end_dt = pd.Timestamp(tail["date"].iloc[-1]).normalize()
    effective_dt = end_dt
    return ma, start_dt, end_dt, effective_dt


def _calc_position_series(total_money_daily: pd.Series) -> pd.Series:
    s = pd.to_numeric(total_money_daily, errors="coerce").dropna()
    if AVG_INCLUDES_TODAY:
        avg_series = s.rolling(WINDOW_SIZE).mean()
    else:
        avg_series = s.shift(1).rolling(WINDOW_SIZE).mean()

    confirm_series = pd.Series(True, index=s.index)
    if PRICE_CONFIRM_MODE == "ma":
        start_ymd = (s.index.min() - pd.Timedelta(days=30)).strftime("%Y%m%d")
        end_ymd = s.index.max().strftime("%Y%m%d")
        etf_daily = _fetch_etf_daily_close(ETF_SYMBOL, start_ymd, end_ymd).set_index("date").sort_index()
        ma_series = etf_daily["close"].shift(1).rolling(MA_WINDOW).mean()
        confirm_series = (etf_daily["close"] > ma_series).reindex(s.index).fillna(False)

    pos = 0
    out = []
    for d in s.index:
        g = float(s.loc[d])
        i = float(avg_series.loc[d]) if pd.notna(avg_series.loc[d]) else float("nan")
        confirm_ok = bool(confirm_series.loc[d]) if d in confirm_series.index else True

        if pd.isna(i):
            out.append(pos)
            continue

        if pos == 0:
            if g > i * VOL_MULTIPLIER and confirm_ok:
                pos = 1
        else:
            if g <= i:
                pos = 0
        out.append(pos)

    return pd.Series(out, index=s.index, name="position_v2").astype(int)


def _extract_args(message_str: str, command: str) -> list[str]:
    tokens = [t for t in str(message_str or "").strip().split() if t.strip()]
    if not tokens:
        return []
    head = tokens[0].lstrip("/").strip().lower()
    if head == str(command).strip().lower():
        return tokens[1:]
    return tokens


def _parse_asof(arg: str | None) -> pd.Timestamp | None:
    if arg is None:
        return None
    s = str(arg).strip()
    if s == "" or s.lower() in {"now", "today", "实时", "今日"}:
        return None
    try:
        if len(s) == 8 and s.isdigit():
            return pd.Timestamp(dt.datetime.strptime(s, "%Y%m%d").date())
    except Exception:
        pass
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"日期无法解析: {s}")
    return pd.Timestamp(ts).normalize()


def _format_lines(lines: Iterable[str]) -> str:
    return "\n".join([str(x) for x in lines if str(x) != ""]).strip()


def etf_v2_signal_text(as_of: pd.Timestamp | None) -> str:
    _disable_akshare_tqdm()
    now = dt.datetime.now()
    today = pd.Timestamp(now.date())

    use_realtime = as_of is None
    asof_dt = today if use_realtime else pd.Timestamp(as_of).normalize()

    spot_map: dict[str, float] = {}
    cur_money = float("nan")

    end_dt = (today - pd.Timedelta(days=1)).normalize() if use_realtime else asof_dt
    n_needed = WINDOW_SIZE + 1
    total_money_all = _fetch_total_money_series_exch(end_dt=end_dt, n_days=n_needed)
    amount_multiplier = 1.0

    if use_realtime:
        if total_money_all.shape[0] < WINDOW_SIZE:
            raise ValueError("数据不足，无法计算信号")
        spot_map = _fetch_index_spot_amount_sina(INDEX_SYMBOLS)
        cur_money = float(sum(spot_map.values()))
        total_money_hist = total_money_all.sort_index()
        effective_asof = None
    else:
        available_dates = total_money_all.index[total_money_all.index <= asof_dt]
        if available_dates.empty:
            raise ValueError("数据不足，无法计算信号")
        effective_asof = pd.Timestamp(available_dates.max()).normalize()
        if effective_asof not in total_money_all.index:
            raise ValueError("数据不足，无法计算信号")
        cur_money = float(total_money_all.loc[effective_asof])
        total_money_hist = total_money_all[total_money_all.index < effective_asof].sort_index()
        if total_money_hist.shape[0] < WINDOW_SIZE:
            raise ValueError("数据不足，无法计算信号")
        spot_map = {
            "SSE": float(_fetch_sse_stock_amount_yuan(effective_asof.strftime("%Y%m%d"))),
            "SZSE": float(_fetch_szse_stock_amount_yuan(effective_asof.strftime("%Y%m%d"))),
        }

    if AVG_INCLUDES_TODAY:
        if use_realtime:
            if WINDOW_SIZE <= 1:
                avg_money = float(cur_money)
            else:
                hist_part = total_money_hist.tail(WINDOW_SIZE - 1)
                avg_money = float(pd.concat([hist_part, pd.Series([cur_money], index=[today])]).mean())
        else:
            avg_money = float(total_money_all[total_money_all.index <= effective_asof].tail(WINDOW_SIZE).mean())
    else:
        avg_money = float(total_money_hist.tail(WINDOW_SIZE).mean())
    vol_ratio = float(cur_money / avg_money) if avg_money > 0 else float("nan")

    pos_series = _calc_position_series(total_money_hist)
    prior_position = int(pos_series.iloc[-1]) if not pos_series.empty else 0

    buy_signal = cur_money > avg_money * VOL_MULTIPLIER
    sell_signal = cur_money <= avg_money

    if prior_position == 0:
        action = "买入" if buy_signal else "空仓"
    else:
        action = "卖出" if sell_signal else "持仓"

    ma_asof = today if use_realtime else (effective_asof if effective_asof is not None else asof_dt)
    try:
        start_ymd = (ma_asof - pd.Timedelta(days=max(120, int(MA_WINDOW) * 4))).strftime("%Y%m%d")
        end_ymd = ma_asof.strftime("%Y%m%d")
        etf_daily_for_ma = _fetch_etf_daily_close(ETF_SYMBOL, start_ymd, end_ymd)
        ma_val, ma_start, ma_end, ma_effective = _calc_ma_price(etf_daily_for_ma, ma_asof, int(MA_WINDOW))
    except Exception:
        ma_val, ma_start, ma_end, ma_effective = None, None, None, None

    lines: list[str] = []
    lines.append(f"标的: {ETF_SYMBOL}")
    lines.append(f"时间: {now:%Y-%m-%d %H:%M:%S}")
    if not use_realtime and effective_asof is not None:
        lines.append(f"回放日期: {effective_asof:%Y-%m-%d}")
    lines.append(f"历史成交额单位修正: * {amount_multiplier:g}")
    if ma_val is None or ma_start is None or ma_end is None:
        lines.append(f"MA{int(MA_WINDOW)}: --")
    else:
        lines.append(f"MA{int(MA_WINDOW)}: {ma_val:.3f}  ({ma_start:%Y-%m-%d} ~ {ma_end:%Y-%m-%d})")

    lines.append("全A成交额(截至当前):")
    if use_realtime:
        for sym in INDEX_SYMBOLS:
            lines.append(f"  {sym}: {spot_map.get(sym, 0.0):.0f}")
    else:
        for k in ["SSE", "SZSE"]:
            if k in spot_map:
                lines.append(f"  {k}: {spot_map.get(k, 0.0):.0f}")
    lines.append(f"  合计: {cur_money:.0f}")
    lines.append(f"窗口判定额: {avg_money:.0f}")
    lines.append(f"放量比: {vol_ratio:.3f}")
    lines.append(f"昨日仓位(按策略回放): {prior_position}")
    lines.append(f"买入信号 => {buy_signal}")
    lines.append(f"卖出信号 => {sell_signal}")
    lines.append(f"今日动作: {action}")

    return _format_lines(lines)


@register("helloworld", "YourName", "一个简单的 Hello World 插件", "1.0.0")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    async def initialize(self):
        _disable_akshare_tqdm()

    @filter.command("helloworld")
    async def helloworld(self, event: AstrMessageEvent):
        user_name = event.get_sender_name()
        yield event.plain_result(f"Hello, {user_name}!")

    @filter.command("etf")
    async def etf(self, event: AstrMessageEvent):
        args = _extract_args(getattr(event, "message_str", ""), "etf")
        date_arg = args[0] if len(args) >= 1 else None
        try:
            as_of = _parse_asof(date_arg)
        except Exception as ex:
            yield event.plain_result(f"参数错误: {ex!r}\n用法: /etf 或 /etf 2026-04-01 或 /etf 20260401")
            return

        try:
            text = await asyncio.to_thread(etf_v2_signal_text, as_of)
        except Exception as ex:
            logger.exception(ex)
            yield event.plain_result(f"运行失败: {ex!r}")
            return
        yield event.plain_result(text)

    async def terminate(self):
        return
