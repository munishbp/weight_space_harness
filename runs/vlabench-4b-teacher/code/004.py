rows = trial("smoke", PRACTICE[:8])
print({k: (v if not isinstance(v, str) else v[:300]) for k, v in rows[0].items()})
