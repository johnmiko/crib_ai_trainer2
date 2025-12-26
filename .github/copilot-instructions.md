Activate the virtual environment with .\.venv\Scripts\Activate.ps1 before doing any work

- Follow standard python best practices
- __init__.py files should be empty
- Always follow the single responsibility principle, where functions should only do one thing
- Use pytest for testing
- Imports always go at the top of the file and are absolute imports, do not use relative imports

- If you ever write any code that involves calculations, make sure to write a unit test to test that the calculation is correct
- When you are done completing any task, run the projects test suite to confirm that none of the existing functionality is broken
Once you have written the test, run the test iteratively and fix things until it passes

use log statements instead of print statements (logger = getLogger(__name__))

The root directory of the repo should contain as little files as possible. Put tests file in a "tests" directory, code files into subdirectory "crib_ai_trainer", trained models into the directory "trained_models" and model definitions into "models" folder. 
If a file starts becoming long (longer than 500 lines), check if some of the code can be refactored into functions and put the functions into other files. If the functions do not group together well, add them to utils.py

 Any tests or scripts should be runnable from the root directory