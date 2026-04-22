import pandas as pd
from pfas_birthweight.pipeline import load_ucmr5

ucmr5 = load_ucmr5()
print("Shape:", ucmr5.shape)
print("\nColumns:", ucmr5.columns.tolist())
print("\nContaminant unique values:")
print(ucmr5["Contaminant"].unique())
print("\nSample rows:")
print(ucmr5.head())