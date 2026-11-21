---
name: verify-m1triton
description: >-
  Drive and verify m1triton end-to-end in a checkout-isolated environment, capture runtime evidence under .audit/, and audit Feature Map contracts.
---

# Verify m1triton

Run these commands from the repository root before claiming a change to m1triton works.

## Workflow

1. `node .agents/skills/verify-m1triton/scripts/driver.mjs doctor`: verify toolchain and checkout-isolated port and state directory.
2. `node .agents/skills/verify-m1triton/scripts/driver.mjs drive`: run the target command against isolated state.
3. `node .agents/skills/verify-m1triton/scripts/driver.mjs evidence`: write timestamped stdout, stderr, and exit code to `.audit/`.
4. `node .agents/skills/verify-m1triton/scripts/audit-feature-map.mjs`: verify every Feature Map file under `.agents/skills/verify-m1triton/references/features/` points to real files and keeps required sections.
5. `node .agents/skills/verify-m1triton/scripts/driver.mjs cleanup`: delete temporary runtime artifacts.
