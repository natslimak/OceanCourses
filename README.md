# OceanCourses

Repository for oceanography data analysis coursework using Python.

## Setup

### 1. Create Virtual Environment
```powershell
python -m venv .ocean
```

### 2. Activate Virtual Environment
```powershell
.ocean\Scripts\Activate.ps1
```

If you encounter an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install Dependencies
```powershell
pip install --upgrade pip
pip install numpy pandas netCDF4 matplotlib
```

## Project Structure

```
OceanCourses/
├── DigitalOcean/
│   └── Assignment_1.ipynb    # Ocean data analysis notebook
├── .ocean/                    # Virtual environment (ignored by git)
├── .gitignore
└── README.md
```

## Assignments

### Assignment 1
- Work with NetCDF ocean data files
- Visualize temperature data (thetao) across depth, latitude, and longitude
- Plot using matplotlib and analyze specific data vectors

## Usage

Open Jupyter notebooks in VS Code or run:
```powershell
jupyter notebook
```

## Requirements

- Python 3.x
- numpy
- pandas
- netCDF4
- matplotlib
