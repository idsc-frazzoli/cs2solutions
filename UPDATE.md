**How to update the package (for developers)**
On Windows:
- Make your changes in the resp. folders
- Make sure 'build' is installed, install using 'py -m pip install --upgrade build'
- Build the Python package by running 'py -m build' in the pyproject.toml folder
- Force reinstall the package (at least for now) %pip install ...\cs2solutions\dist\cs2solutions-0.0.5.2-py3-none-any.whl --force-reinstall Make sure you run this command at the beginning of the notebook
- If a newly implemented function isn't recognised try resetting your Python kernel

On Linux:
- Make your changes in the resp. folders
- Make sure 'build' is installed, install using 'python3 -m pip install --upgrade build'
- Build the Python package by running 'python3 -m build' in the pyproject.toml folder
- Force reinstall the package (at least for now) %pip install ...\cs2solutions\dist\cs2solutions-0.0.5.2-py3-none-any.whl --force-reinstall Make sure you run this command at the beginning of the notebook
- If a newly implemented function isn't recognised try resetting your Python kernel