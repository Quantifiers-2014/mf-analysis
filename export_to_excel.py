"""Copy the data in data/arbitrage.db into an Excel file you can open and browse.

Run from the project folder, with the virtual environment switched on:
    python export_to_excel.py
"""

import sqlite3  # built into Python: reads SQLite database files

import pandas as pd  # the table library this project uses

# Open the database file.
conn = sqlite3.connect("data/arbitrage.db")

# Read each table into a DataFrame (pandas' name for a table).
funds = pd.read_sql("SELECT * FROM funds", conn)
nav = pd.read_sql("SELECT * FROM nav", conn)
flags = pd.read_sql("SELECT * FROM validation_flags", conn)
conn.close()

# Make NAV easier to read: one row per date, one column per fund.
names = dict(zip(funds["amfi_code"], funds["scheme_name"].str.split(" - ").str[0], strict=True))
nav["fund"] = nav["amfi_code"].map(names)
nav_wide = nav.pivot(index="date", columns="fund", values="nav").sort_index(ascending=False)

# Write each table to its own sheet in one Excel file.
with pd.ExcelWriter("data/arbitrage_data.xlsx") as writer:
    funds.to_excel(writer, sheet_name="Funds", index=False)
    nav_wide.to_excel(writer, sheet_name="NAV by date")
    flags.to_excel(writer, sheet_name="Data flags", index=False)

counts = f"{len(funds)} funds, {len(nav_wide)} dates, {len(flags)} flags"
print(f"Saved data/arbitrage_data.xlsx: {counts}")
