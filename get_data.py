from ucimlrepo import fetch_ucirepo

ds = fetch_ucirepo(id=94)
df = ds.data.features.copy()
df["is_spam"] = ds.data.targets.values.ravel()
df.to_csv("spambase.csv", index=False)
print(df.shape)