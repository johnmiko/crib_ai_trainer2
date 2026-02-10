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

- You have generic permission to run powershell, python, pytest, and grep (whatever the short command for it is) as well as write access to files

Activate the virtual environment with .\.venv\Scripts\Activate.ps1 before doing any work
- If you ever write any code that involves calculations, make sure to write a unit test to test that the calculation is correct
- When you are done completing any task, run the projects test suite to confirm that none of the existing functionality is broken
Once you have written the test, run the test iteratively and fix things until it passes
If you are coding in python, use log statements instead of print statements (logger = getLogger(__name__))
never use fallbacks unless I specifically ask you to. The rate at which you add fallbacks is too much. It's more likely to introduce unwanted small bugs rather than help because we want things to hard fail usually if they are not working so we know they are broken. We don't want things to fail silently so it doesn't break, and it looks like it's working but we don't notice/can't tell. Yes sometimes fallbacks are good for backwards compatibility or in the case where it's possible a feature may not load or do something correctly. But in the case of a text description a fallback is completely unnecessary