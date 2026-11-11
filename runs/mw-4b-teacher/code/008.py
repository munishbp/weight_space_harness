for margin in (8.0, 12.0, 16.0):
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", KB[:10], values_for(SB[:10], margin), PROTECT[L], 1.0)
    print(f"fit on button seeds 0-9 | L{L} | margin {margin} | relative change {e['relative_change']:.4f} | weights changed {e['changed_weights_pct']:.1f}% of the matrix")
    p_move("   edited")
    tools.reset_edit()
