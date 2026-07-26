# Step-2 QP isolation test — report

**Branch:** `claude/fk-bypass-aware-tuning`

**Test scenario:** `start_a=3, start_b=3, n_steps=1` — system initialised at (3, 3) docked configuration; swing arm 'b' commanded to anchor 4. **No prior drift from steps 0/1.**

## Side-by-side metrics

| metric | A: bypass=ON (current default) | B: bypass=OFF (FK ref to QP) |
|---|---|---|
