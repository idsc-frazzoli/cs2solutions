**How to update the package (for developers)**
On Windows:
- Download the package from GitHub as a .zip file and extract it
- Make your changes in the resp. folders
- Make sure 'build' is installed, install using 'py -m pip install --upgrade build'
- Build the Python package by running 'py -m build' in the pyproject.toml folder
- Force reinstall the package (at least for now) %pip install ...\cs2solutions\dist\cs2solutions-x.x.x.x-py3-none-any.whl --force-reinstall Make sure you run this command at the beginning of the notebook
- If a newly implemented function isn't recognised try resetting your Python kernel

On Linux:
- Download the package from GitHub as a .zip file and extract it
- Make your changes in the resp. folders
- Make sure 'build' is installed, install using 'python3 -m pip install --upgrade build'
- Build the Python package by running 'python3 -m build' in the pyproject.toml folder
- Force reinstall the package (at least for now) %pip install ...\cs2solutions\dist\cs2solutions-x.x.x.x-py3-none-any.whl --force-reinstall Make sure you run this command at the beginning of the notebook
- If a newly implemented function isn't recognised try resetting your Python kernel