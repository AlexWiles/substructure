---
name: unit-conversion
description: Convert measurements between metric and imperial units (km/mi, kg/lb). Use when the user asks to convert a measurement or mentions units.
---

# Unit conversion

Loading this skill unlocks the `convert_units` tool.

## Steps

1. Identify the value and the source and target units.
2. Call `convert_units` with `{ value, from, to }` using unit codes: `km`, `mi`, `kg`, `lb`.
3. Report the result, rounded sensibly for the context.

Do not do the arithmetic yourself — call the tool so conversions stay exact.
