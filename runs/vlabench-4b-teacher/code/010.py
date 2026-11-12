r = plan_small[0]
print([tok.decode([i]) for i in r["ids"]])
import re
print(re.findall(r"skill[^\n]{0,200}", vp.prompt())[:12])
print(vp.prompt()[:1500])
