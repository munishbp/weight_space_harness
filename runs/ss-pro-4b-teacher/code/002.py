tok = processor.tokenizer
r = base_res[quick[0]["id"]]
print([tok.decode([t]) for t in r["ids"]])
print("digits as tokens:", [tok.encode(str(d), add_special_tokens=False) for d in range(10)], "| '473':", tok.encode("473", add_special_tokens=False), "| ' 473':", tok.encode(" 473", add_special_tokens=False))
