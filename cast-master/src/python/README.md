Cast API with Python
========================

A Python wrapper (pycast) has been created to help with getting programs running more quickly.

Examples:
- **pycaster** a simple command line tool to connect and stream images. support for writing out images using PIL.
- **pysidecaster** a simple Qt based graphical program to connect and stream/view images. uses PySide2 for usage of the Qt libraries.

Executing under Linux:
- Install Pillow (latest PIL library) and PySide2 using pip
- Copy pycast.so and libcast.so to {cast_api_path}/src/python
- Execute: LD_LIBRARY_PATH=. python3 {clarius_python_example}.py

Executing under Windows10 (Python3.8, not working properly)
- Copy cast.dll, cast.lib and pycast.pyd to {cast_api_path}/src/python
- Turn on Mobile hotspot: settings -> Network & Internet -> Mobile hotspot
- Connect mobile device and Clarius probe to Windows Mobile hotspot
- Execute: python {clarius_python_example}.py --address $ip$ -- port $port$