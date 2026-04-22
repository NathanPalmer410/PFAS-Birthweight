from pfas_birthweight.pipeline import load_ucmr5

ucmr5 = load_ucmr5()
ucmr5["detected"] = ucmr5["result"] > (ucmr5["MRL"] / 2)
print("Total rows:", len(ucmr5))
print("Detected rows:", ucmr5["detected"].sum())
print("Detection rate:", ucmr5["detected"].mean() * 100, "%")
print("\nDetections by state:")
print(ucmr5.groupby("State")["detected"].sum())
print("\nDetections by compound:")
print(ucmr5.groupby("Contaminant")["detected"].sum().sort_values(ascending=False))