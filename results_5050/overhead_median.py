import sys

import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "5050-reference_steps.csv"
overhead = pd.read_csv(path).overhead_us.iloc[1:]

# should be 5787 steps
print(f"median overhead: {overhead.median():.0f} us   ({len(overhead)} steps)")
