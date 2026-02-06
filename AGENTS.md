# AGENTS.md

## Purpose
- The `crib_ai_trainer2` repo exists to build a strong crib AI. All work must respect the rules of cribbage as implemented in the engine.

## Scope
- Scope includes **all** crib-related repos in this workspace: `crib_ai_trainer2`, `crib_engine`, `crib_back`, and `crib_front`.

## Testing Expectations
- **After any code edits**, run a short smoke test (or a small pytest if that’s more appropriate).
- If you’re unsure which test is best, pick the smallest command that exercises the changed code path.

## Communication & Tone
- The user is very sick and has severe brain fog, memory issues, and impaired reading. Keep responses **short, clear, and not overly dense**.
- Be supportive and avoid long, complex explanations. Summarize decisions and next steps plainly.
- Assume the user can follow technical content (IQ ~135), but keep cognitive load low.

## Preferences
- No special safety restrictions beyond standard caution.
- When making code changes, do not include fallbacks
- If you make changes to crib_engine, make sure to run all the tests in crib_engine and crib_back, to check that you didn't break anything

- Change the phrasing of continuation sentences like this "If you want me to verify the final max shard number or list the new filenames, I can."
    to instead be like this "Do you want me to verify the final max shard number or list the new filenames?"

- You have generic permission to run powershell, python, pytest, and grep (whatever the short command for it is)