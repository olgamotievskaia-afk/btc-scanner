"""
Сканер сетапов: снятие ликвидности -> IFVG -> вход по ретесту, фильтр RR>=1.0.
Логика 1-в-1 повторяет проверенный бэктест: 11 активов проверены отдельно
на реальной истории (6-7 лет для большинства, 3+ года для SOL/SUI),
везде Profit Factor >= 2.0.

Каждый запуск (раз в 15 минут через GitHub Actions):
  1. Тянет последние ~200 дней 15м свечей с публичного Binance Futures API
     для каждого символа из списка SYMBOLS.
  2. Строит тренд по 4Ч, свинги на 15м, ищет свип+IFVG, фильтрует по RR.
  3. Сравнивает текущее состояние (пендинг/позиция/ничего) с прошлым
     запуском и шлёт алерт ТОЛЬКО на изменения, отдельно по каждому символу.
  4. Если в одном проходе сигнал пришёл сразу по нескольким символам —
     присылает отдельное предупреждение о кластерном риске (см. README).
"""
import os
import json
import time
import requests
import numpy as np
import pandas as pd

# ---------------- Параметры стратегии (проверено бэктестом на каждом активе) ----------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "HYPEUSDT"]
INTERVAL = "15m"
HISTORY_DAYS = 200
SWING_K = 2
ATR_PERIOD = 14
MAX_SWEEP_DEPTH_ATR = 1.5
RETURN_CONFIRM_BARS = 3
IFVG_SEARCH_BARS = 20
ENTRY_FILL_BARS = 20
STOP_CLOSE_RATIO = 2.0
MIN_RR = 1.0                 # проверено на всех 11 активах: PF 2.0-3.7
RISK_PER_TRADE = 0.01        # только для справки в алерте

STATE_FILE = os.environ.get("STATE_FILE", "state.json")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_HOSTS = ["https://fapi.binance.com", "https://data-api.binance.vision"]


# ---------------- Загрузка данных ----------------
def fetch_klines(symbol, interval, days):
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000
    last_err = None
    for host in BINANCE_HOSTS:
        try:
            all_rows = []
            cursor = start_time
            base = "/fapi/v1/klines" if "fapi" in host else "/api/v3/klines"
            while cursor < end_time:
                resp = requests.get(f"{host}{base}",
                                     params={"symbol": symbol, "interval": interval,
                                             "startTime": cursor, "limit": 1500},
                                     timeout=15)
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                all_rows.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(0.15)
            if all_rows:
                df = pd.DataFrame(all_rows, columns=[
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "qav", "trades", "taker_base", "taker_quote", "ignore"])
                df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = df[c].astype(float)
                df = df[["datetime", "open", "high", "low", "close", "volume"]].drop_duplicates("datetime")
                df = df.sort_values("datetime").reset_index(drop=True)
                return df.iloc[:-1].reset_index(drop=True)  # отбрасываем незакрытую свечу
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Не удалось получить данные {symbol}: {last_err}")


# ---------------- Индикаторы ----------------
def compute_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(),
                     (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_swings(df, k=2):
    high, low = df["high"].values, df["low"].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(k, n - k):
        wh = high[i - k:i + k + 1]
        wl = low[i - k:i + k + 1]
        if high[i] == wh.max() and (wh == wh.max()).sum() == 1:
            sh[i] = True
        if low[i] == wl.min() and (wl == wl.min()).sum() == 1:
            sl[i] = True
    return sh, sl


def build_4h_trend(df15):
    df4h = df15.set_index("datetime").resample("4h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    sh, sl = find_swings(df4h, SWING_K)
    df4h["sh"], df4h["sl"] = sh, sl
    trend = np.zeros(len(df4h), dtype=int)
    swing_highs, swing_lows = [], []
    last_trend = 0
    for i in range(len(df4h)):
        j = i - SWING_K
        if j >= 0:
            if df4h["sh"].iloc[j]:
                swing_highs.append((j, df4h["high"].iloc[j]))
            if df4h["sl"].iloc[j]:
                swing_lows.append((j, df4h["low"].iloc[j]))
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1][1] > swing_highs[-2][1]
            hl = swing_lows[-1][1] > swing_lows[-2][1]
            lh = swing_highs[-1][1] < swing_highs[-2][1]
            ll = swing_lows[-1][1] < swing_lows[-2][1]
            if hh and hl:
                last_trend = 1
            elif lh and ll:
                last_trend = -1
        trend[i] = last_trend
    df4h["trend"] = trend
    return df4h[["datetime", "trend"]]


def attach_trend_to_15m(df15, df4h_trend):
    df15 = df15.sort_values("datetime")
    merged = pd.merge_asof(df15, df4h_trend, on="datetime", direction="backward")
    merged["trend"] = merged["trend"].fillna(0).astype(int)
    return merged


def find_fvg_in_leg(df, start_idx, end_idx, direction):
    best = None
    for i in range(start_idx, end_idx - 1):
        c1_high, c1_low = df["high"].iloc[i], df["low"].iloc[i]
        c3_high, c3_low = df["high"].iloc[i + 2], df["low"].iloc[i + 2]
        if direction == "bearish" and c1_low > c3_high:
            best = (c3_high, c1_low, i + 2)
        elif direction == "bullish" and c1_high < c3_low:
            best = (c1_high, c3_low, i + 2)
    return best


# ---------------- Основной проход по одному символу ----------------
def scan(df):
    sh, sl = find_swings(df, SWING_K)
    df = df.copy()
    df["sh"], df["sl"] = sh, sl
    df["atr"] = compute_atr(df, ATR_PERIOD)
    n = len(df)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    trend, atr, times = df["trend"].values, df["atr"].values, df["datetime"].values

    swing_highs, swing_lows = [], []
    pending = None
    position = None

    i = 0
    while i < n:
        j = i - SWING_K
        if j >= 0:
            if df["sh"].iloc[j]:
                swing_highs.append({"idx": j, "price": high[j], "broken": False})
            if df["sl"].iloc[j]:
                swing_lows.append({"idx": j, "price": low[j], "broken": False})

        if position is not None:
            hit_stop = hit_target = False
            if position["side"] == "long":
                if low[i] <= position["stop"]:
                    hit_stop = True
                elif high[i] >= position["target"]:
                    hit_target = True
            else:
                if high[i] >= position["stop"]:
                    hit_stop = True
                elif low[i] <= position["target"]:
                    hit_target = True
            if hit_stop or hit_target:
                position = None

        if position is None and pending is not None:
            if i > pending["expire_idx"]:
                pending = None
            else:
                filled = (pending["side"] == "long" and low[i] <= pending["limit_price"]) or \
                         (pending["side"] == "short" and high[i] >= pending["limit_price"])
                invalidated = (pending["side"] == "long" and low[i] <= pending["stop"]) or \
                              (pending["side"] == "short" and high[i] >= pending["stop"])
                if invalidated:
                    pending = None
                elif filled:
                    position = {"side": pending["side"], "entry": pending["limit_price"],
                                "entry_time": str(times[i]), "stop": pending["stop"],
                                "target": pending["target"]}
                    pending = None

        if position is None and pending is None and i >= 4:
            cur_trend = trend[i]
            if cur_trend == 1:
                unbroken = [s for s in swing_lows if not s["broken"] and s["idx"] < i]
                if unbroken:
                    lvl = unbroken[-1]
                    if low[i] < lvl["price"]:
                        depth = lvl["price"] - low[i]
                        if not np.isnan(atr[i]) and depth <= MAX_SWEEP_DEPTH_ATR * atr[i]:
                            for r in range(i, min(i + RETURN_CONFIRM_BARS, n)):
                                if close[r] > lvl["price"]:
                                    lvl["broken"] = True
                                    prior_highs = [s for s in swing_highs if s["idx"] < lvl["idx"]]
                                    leg_start = prior_highs[-1]["idx"] if prior_highs else max(0, lvl["idx"] - 20)
                                    fvg = find_fvg_in_leg(df, leg_start, i, "bearish")
                                    if fvg:
                                        zlo, zhi, _ = fvg
                                        for b in range(r, min(r + IFVG_SEARCH_BARS, n)):
                                            if close[b] > zhi:
                                                sweep_extreme = low[i]
                                                ifvg_h = zhi - zlo
                                                gap = zlo - sweep_extreme
                                                stop_px = sweep_extreme * 0.999 if (ifvg_h > 0 and gap <= STOP_CLOSE_RATIO * ifvg_h) else zlo * 0.999
                                                limit_px = zhi
                                                cand = [s for s in swing_highs if not s["broken"] and s["price"] > close[b]]
                                                target_px = min(c["price"] for c in cand) if cand else None
                                                if target_px:
                                                    rr = (target_px - limit_px) / (limit_px - stop_px) if limit_px != stop_px else 0
                                                    if rr >= MIN_RR:
                                                        pending = {"side": "long", "limit_price": limit_px,
                                                                   "stop": stop_px, "target": target_px,
                                                                   "expire_idx": b + ENTRY_FILL_BARS,
                                                                   "setup_time": str(times[b]), "planned_rr": rr}
                                                break
                                    break
            elif cur_trend == -1:
                unbroken = [s for s in swing_highs if not s["broken"] and s["idx"] < i]
                if unbroken:
                    lvl = unbroken[-1]
                    if high[i] > lvl["price"]:
                        depth = high[i] - lvl["price"]
                        if not np.isnan(atr[i]) and depth <= MAX_SWEEP_DEPTH_ATR * atr[i]:
                            for r in range(i, min(i + RETURN_CONFIRM_BARS, n)):
                                if close[r] < lvl["price"]:
                                    lvl["broken"] = True
                                    prior_lows = [s for s in swing_lows if s["idx"] < lvl["idx"]]
                                    leg_start = prior_lows[-1]["idx"] if prior_lows else max(0, lvl["idx"] - 20)
                                    fvg = find_fvg_in_leg(df, leg_start, i, "bullish")
                                    if fvg:
                                        zlo, zhi, _ = fvg
                                        for b in range(r, min(r + IFVG_SEARCH_BARS, n)):
                                            if close[b] < zlo:
                                                sweep_extreme = high[i]
                                                ifvg_h = zhi - zlo
                                                gap = sweep_extreme - zhi
                                                stop_px = sweep_extreme * 1.001 if (ifvg_h > 0 and gap <= STOP_CLOSE_RATIO * ifvg_h) else zhi * 1.001
                                                limit_px = zlo
                                                cand = [s for s in swing_lows if not s["broken"] and s["price"] < close[b]]
                                                target_px = max(c["price"] for c in cand) if cand else None
                                                if target_px:
                                                    rr = (limit_px - target_px) / (stop_px - limit_px) if limit_px != stop_px else 0
                                                    if rr >= MIN_RR:
                                                        pending = {"side": "short", "limit_price": limit_px,
                                                                   "stop": stop_px, "target": target_px,
                                                                   "expire_idx": b + ENTRY_FILL_BARS,
                                                                   "setup_time": str(times[b]), "planned_rr": rr}
                                                break
                                    break
        i += 1

    return position, pending, int(trend[-1])


# ---------------- Telegram ----------------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[WARN] TELEGRAM_TOKEN / TELEGRAM_CHAT_ID не заданы:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    resp = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}, timeout=15)
    if resp.status_code != 200:
        print("[ERROR] Telegram:", resp.text)


def fmt_setup(symbol, side, entry, stop, target, rr):
    dirn = "LONG (покупка)" if side == "long" else "SHORT (продажа)"
    return (f"<b>{symbol}: {dirn}</b>\n"
            f"Вход (лимит, ретест IFVG): {entry:.6g}\n"
            f"Стоп: {stop:.6g}\n"
            f"Тейк: {target:.6g}\n"
            f"Плановый RR: {rr:.2f}\n"
            f"Риск 1% депозита -> объём = (0.01 * депозит) / {abs(entry-stop):.6g}")


# ---------------- main ----------------
def main():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            all_prev = json.load(f)
    else:
        all_prev = {}

    all_new_state = {}
    new_setups_this_run = []

    for symbol in SYMBOLS:
        prev = all_prev.get(symbol, {"pending": None, "position": None})
        try:
            df = fetch_klines(symbol, INTERVAL, HISTORY_DAYS)
            df4h_trend = build_4h_trend(df)
            merged = attach_trend_to_15m(df, df4h_trend)
            position, pending, trend = scan(merged)
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            all_new_state[symbol] = prev  # оставляем прошлое состояние, не теряем его
            continue

        had_pending = prev.get("pending") is not None
        had_position = prev.get("position") is not None

        if pending is not None and not had_pending and position is None:
            send_telegram(fmt_setup(symbol, pending["side"], pending["limit_price"],
                                     pending["stop"], pending["target"], pending["planned_rr"]))
            new_setups_this_run.append(symbol)
        if position is not None and not had_position:
            side_txt = "LONG" if position["side"] == "long" else "SHORT"
            send_telegram(f"<b>{symbol}: вход исполнен {side_txt}</b>\n"
                           f"Цена входа: {position['entry']:.6g}\nСтоп: {position['stop']:.6g}\n"
                           f"Тейк: {position['target']:.6g}")
        if had_position and position is None and pending is None:
            send_telegram(f"<b>{symbol}: позиция закрыта</b> (стоп либо тейк — сверься с графиком).")
        if had_pending and pending is None and position is None and not prev.get("position"):
            send_telegram(f"{symbol}: сетап отменён (не заполнился либо стоп раньше входа).")

        all_new_state[symbol] = {"pending": pending, "position": position, "trend": trend}
        print(f"{symbol}: trend={trend} pending={bool(pending)} position={bool(position)}")

    # предупреждение о кластерном риске, если сигналы пришли сразу по нескольким активам
    if len(new_setups_this_run) >= 2:
        send_telegram(f"⚠️ <b>Кластерный сигнал</b>: новый сетап одновременно по {len(new_setups_this_run)} "
                       f"активам ({', '.join(new_setups_this_run)}). Это, вероятно, одно и то же движение "
                       f"рынка, а не {len(new_setups_this_run)} независимых сигналов — раздели обычный риск "
                       f"на сделку между ними, не бери полный риск на каждый.")

    with open(STATE_FILE, "w") as f:
        json.dump(all_new_state, f, indent=2, default=str)


if __name__ == "__main__":
    main()
