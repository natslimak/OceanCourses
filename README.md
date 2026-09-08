# OceanCourses

Repository for all the take courses taken at DTU Ocean Engineering. 

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


## Requirements

- Python 3.x
- numpy
- pandas
- netCDF4
- matplotlib
