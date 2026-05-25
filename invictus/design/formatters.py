"""
invictus.design.formatters
==========================
Number / currency / percentage formatting helpers.
"""


def fmt_currency(val: float, decimals: int = 0) -> str:
    """`$1,234` or `-$1,234`. Negative values get a leading minus, not parens."""
    return f"${val:,.{decimals}f}" if val >= 0 else f"-${abs(val):,.{decimals}f}"


def fmt_pct(val: float, decimals: int = 2, signed: bool = False) -> str:
    """`+1.23%` / `-1.23%` / `1.23%`."""
    sign = "+" if signed and val > 0 else ""
    return f"{sign}{val * 100:.{decimals}f}%"


def fmt_signed_currency(val: float, decimals: int = 0) -> str:
    """`+$1,234` / `-$1,234`."""
    sign = "+" if val >= 0 else "-"
    return f"{sign}${abs(val):,.{decimals}f}"
