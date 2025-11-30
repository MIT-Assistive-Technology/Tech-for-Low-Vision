# Tech-for-Low-Vision

Project for visualizing 2D cross sections of 3D models

To run, navigate to the root directory

Install Python packages:

```
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

To clean the image cache, run:

```
python3 slicer.py clean
```

To run the program, run:

```
python3 app.py
```

and navigate to http://127.0.0.1:3000