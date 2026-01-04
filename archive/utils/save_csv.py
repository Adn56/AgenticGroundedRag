# src/utils/save_csv.py
import csv

def save_csv(df, path):
    df.to_csv(
        path,
        index=False,
        sep=",",
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )
