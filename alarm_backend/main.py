from fastapi import FastAPI, HTTPException
from pvcI_files import list_pvc_files, read_pvc_file, read_all_pvc_files
from pvcI_health_monitor import (
    compute_pvcI_file_health,
    compute_pvcI_overall_health,
    compute_pvcI_overall_health_weighted,
    HealthConfig,
)
from config import PVCI_FOLDER
import os

app = FastAPI(title="Plant Alarm Data System", version="1.0")

@app.get("/all-pvci-files")
def get_all_pvc_files():
    """Return list of all PVC-I plant files"""
    try:
        files = list_pvc_files()
        return {"total_files": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pvc-file/{filename}")
def get_pvc_file_data(filename: str):
    """Return metadata + first 10 rows of a specific PVC-I file"""
    try:
        file_data = read_pvc_file(filename)
        return file_data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {filename} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pvcI-files-data")
def get_all_pvc_files_data():
    """Return data from all PVC-I files, including first 10 rows of each file"""
    try:
        files_data = read_all_pvc_files()
        return files_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pvcI-health/overall")
def get_pvcI_overall_health(
    bin_size: str = "10T",
    alarm_threshold: int = 10,
    max_workers: int = 4,
    per_file_timeout: int | None = 30
):
    """Get overall health metrics for all files"""
    try:
        config = HealthConfig(bin_size=bin_size, alarm_threshold=alarm_threshold)
        return compute_pvcI_overall_health(PVCI_FOLDER, config, max_workers, per_file_timeout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pvcI-health/overall/weighted")
def get_pvcI_overall_health_weighted_endpoint(
    bin_size: str = "10T",
    alarm_threshold: int = 10,
    max_workers: int = 4,
    per_file_timeout: int | None = 30
):
    """Get overall health metrics (weighted by total bins across all files)"""
    try:
        config = HealthConfig(bin_size=bin_size, alarm_threshold=alarm_threshold)
        return compute_pvcI_overall_health_weighted(PVCI_FOLDER, config, max_workers, per_file_timeout)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pvcI-health/{filename}")
def get_pvcI_file_health_endpoint(filename: str):
    """Get health metrics for a specific file"""
    try:
        # Validate filename
        if not filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        file_path = os.path.join(PVCI_FOLDER, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Log file access
        print(f"Processing file: {file_path}")
        
        result = compute_pvcI_file_health(file_path)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
