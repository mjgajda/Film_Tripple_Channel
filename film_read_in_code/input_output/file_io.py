# io/file_io.py
import pandas as pd
from tkinter import Tk, filedialog

def save_dataframe_as_csv(df: pd.DataFrame, default_name="output.csv") -> str:
    root = Tk()
    root.withdraw()
    fpath = filedialog.asksaveasfilename(
        title="Save CSV file",
        defaultextension=".csv",
        initialfile=default_name,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    root.destroy()

    if not fpath:
        raise KeyboardInterrupt("User cancelled save.")

    df.to_csv(fpath, index=False)
    print(f"✅ DataFrame successfully saved to: {fpath}")
    return fpath
