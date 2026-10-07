# ---------------- Счета (ПРАВЬ ВРУЧНУЮ: код не видит счета сам) ----------------
# balance - текущий баланс, risk - доля риска на сделку, max_lev - макс. плечо на платформе
# input "btc" - объём вводится в BTC (Hash Hedge), иначе в USD маржи (Upscale)
ACCOUNTS = [
    {"name": "Hash Hedge 10K", "balance": 10000.0, "risk": 0.005, "max_lev": 5.0, "input": "btc"},
    {"name": "Upscale 1K", "balance": 1048.5, "risk": 0.01, "max_lev": 5.0},
    {"name": "Upscale 5K", "balance": 4658.54, "risk": 0.005, "max_lev": 5.0},
]


def fmt_setup(symbol, side, entry, stop, target, rr):
    dirn = "LONG (покупка)" if side == "long" else "SHORT (продажа)"
    coin = symbol.replace("USDT", "")
    stop_dist = abs(entry - stop)

    lines = [f"<b>{symbol}: {dirn}</b>",
             f"Вход (лимит, ретест IFVG): {entry:.6g}",
             f"Стоп: {stop:.6g}",
             f"Тейк: {target:.6g}",
             f"Плановый RR: {rr:.2f}",
             ""]

    for acc in ACCOUNTS:
        risk_amt = acc["balance"] * acc["risk"]
        qty_risk = risk_amt / stop_dist if stop_dist > 0 else 0
        qty_cap = acc["max_lev"] * acc["balance"] / entry
        qty = min(qty_risk, qty_cap)
        notional = qty * entry
        lev = int(min(acc["max_lev"], max(1, np.ceil(round(notional / acc["balance"], 6)))))
        margin = notional / lev
        real_risk = qty * stop_dist
        cap_note = " (упёрлись в потолок плеча)" if qty_cap < qty_risk else ""
        if acc.get("input") == "btc":
            how = f"<b>{qty:.4f} {coin}</b> (поле «Сумма», плечо 5X)"
        else:
            how = f"<b>{qty:.4f} {coin}</b>, плечо x{lev}, маржа ≈ ${margin:,.0f}"
        lines.append(f"{acc['name']}: {how}, риск ≈ ${real_risk:,.2f}{cap_note}")

    lines.append("")
    lines.append("Перед отправкой: «Размер позиции» в панели справа = объём выше; "
                 "Стоп и Тейк введи ЦЕНОЙ (кнопка Price), не None.")
    return "\n".join(lines)
