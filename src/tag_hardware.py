import io
import sys
import re
import pandas as pd
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

MASTER_FILE = Path("data/master_games.csv")

HARDWARE_PATTERNS = [
    r"\bcontroller\b",
    r"\bwii remote\b",
    r"\bwii nunchuk\b",
    r"\bwii nunchuck\b",
    r"\bnunchuk\b",
    r"\bnunchuck\b",
    r"\bwavebird\b",
    r"\bsensor bar\b",
    r"\bwii speak\b",
    r"\bheadset\b",
    r"\bmemory card\b",
    r"\brumble pak\b",
    r"\btransfer pak\b",
    r"\bexpansion pak\b",
    r"\bstylus pack\b",
    r"\bscreen protector\b",
    r"\bcarrying case\b",
    r"\btravel kit\b",
    r"\bcharging dock\b",
    r"\bcharging cradle\b",
    r"\bcables?\b",
    r"\bac adapter\b",
    r"\bpower supply\b",
    r"\brf switch\b",
    r"\b(?:wii|ds|dsi|3ds|gba|ac|car|usb|wireless)\s+charger\b",
    r"\bcharger\b(?!\s*vs\.)",
    r"\bgameshark\b",
    r"\bgame shark\b",
    r"\baction replay\b",
    r"\bgame boy advance sp\b",
    r"\bgame boy micro\b",
    r"\bgame boy color\b(?=\s*[-\[])",
    r"\bds lite\b",
    r"\bnintendo dsi\b",
    r"\bdsi xl\b",
    r"\bnintendo 3ds\b(?=\s*[-\[])",
    r"\b3ds xl\b",
    r"\bnew nintendo 3ds\b",
    r"\bnew 3ds\b",
    r"\b2ds xl\b",
    r"\bnew 2ds\b",
    r"\bnintendo 2ds\b",
    r"\bwii u pro controller\b",
    r"\bwii u gamepad\b",
    r"\bwii u nunchuk\b",
]

_HARDWARE_RE = re.compile(
    "|".join(f"(?:{p})" for p in HARDWARE_PATTERNS),
    re.IGNORECASE,
)


def is_hardware_title(title: str) -> bool:
    return bool(_HARDWARE_RE.search(str(title)))


def main():
    if not MASTER_FILE.exists():
        print(f"ERROR: {MASTER_FILE} not found. Run the pipeline first.")
        return

    master = pd.read_csv(MASTER_FILE, low_memory=False)
    print(f"Loaded {len(master):,} rows from {MASTER_FILE}\n")

    master["is_hardware"] = master["title"].apply(is_hardware_title)

    n_hardware = master["is_hardware"].sum()
    print(f"Tagged {n_hardware:,} rows as hardware ({n_hardware/len(master):.1%} of catalog)\n")

    breakdown = (
        master[master["is_hardware"]]
        .groupby("console")
        .size()
        .sort_values(ascending=False)
    )
    print("Hardware by console:")
    for console, count in breakdown.items():
        pct = count / len(master[master["console"] == console])
        print(f"  {console:<22} {count:>4}  ({pct:.1%} of that console's catalog)")

    print("\n--- Flagged titles (sample, first 10 per console) ---")
    for console in master["console"].unique():
        hw = master[(master["is_hardware"]) & (master["console"] == console)]
        if hw.empty:
            continue
        print(f"\n{console} ({len(hw)}):")
        for t in sorted(hw["title"].tolist())[:10]:
            print(f"  {t}")
        if len(hw) > 10:
            print(f"  ... and {len(hw)-10} more")

    master.to_csv(MASTER_FILE, index=False)
    print(f"\nSaved updated master_games.csv with is_hardware column")


if __name__ == "__main__":
    main()
