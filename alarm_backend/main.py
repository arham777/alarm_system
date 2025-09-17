from fastapi import FastAPI, HTTPException
from pvcI_files import list_pvc_files, read_pvc_file, read_all_pvc_files

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
