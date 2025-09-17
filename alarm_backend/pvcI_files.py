import os
import pandas as pd
from typing import Dict, Any, List
from config import PVCI_FOLDER

def list_pvc_files() -> List[str]:
    """Return all CSV files from PVC-I folder"""
    files = [f for f in os.listdir(PVCI_FOLDER) if f.lower().endswith(".csv")]
    return files

def read_all_pvc_files(max_rows: int = 10) -> Dict[str, Any]:
    """Read all PVC-I files and return their data"""
    files = list_pvc_files()
    all_files_data = []
    
    for filename in files:
        try:
            file_data = read_pvc_file(filename)
            all_files_data.append(file_data)
        except Exception as e:
            print(f"Error reading file {filename}: {str(e)}")
            continue
    
    return {
        "total_files": len(files),
        "files_data": all_files_data
    }

def read_pvc_file(filename: str, max_rows: int = 10) -> Dict[str, Any]:
    """Read a specific PVC-I file and return data with standardized columns"""
    file_path = os.path.join(PVCI_FOLDER, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {filename} not found.")

    # Define expected columns
    expected_columns = [
        "Event Time", "Location Tag", "Source", "Condition", 
        "Action", "Priority", "Description", "Value", "Units"
    ]

    try:
        # Try reading with comma separator first since that appears to be the standard
        df = pd.read_csv(
            file_path, 
            sep=',',
            encoding='utf-8', 
            engine='python',  # Using python engine for better quote handling
            skipinitialspace=True,
            quotechar='"',    # Explicitly set quote character
            on_bad_lines='skip',
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skiprows=lambda x: x < 8  # Skip first 8 rows which contain metadata
        )
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # If columns don't match expected, try tab separator
        if not all(col in df.columns for col in expected_columns):
            df = pd.read_csv(
                file_path, 
                sep='\t',
                encoding='utf-8', 
                engine='python',
                skipinitialspace=True,
                quotechar='"',    # Explicitly set quote character
                on_bad_lines='skip',
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                skiprows=lambda x: x < 8  # Skip first 8 rows which contain metadata
            )
            df.columns = df.columns.str.strip()
        
        # If we still don't have the right columns, try reading the whole file
        if not all(col in df.columns for col in expected_columns):
            df = pd.read_csv(
                file_path, 
                sep=',',
                encoding='utf-8', 
                engine='python',
                skipinitialspace=True,
                quotechar='"',    # Explicitly set quote character
                on_bad_lines='skip',
                dtype=str,
                keep_default_na=False,
                na_filter=False
            )
            df.columns = df.columns.str.strip()
        
        # Ensure we have all required columns
        for col in expected_columns:
            if col not in df.columns:
                df[col] = ""
        
        # Select only the columns we want, in the right order
        df = df[expected_columns]
        
        # Replace any remaining whitespace-only cells with empty string
        df = df.replace(r'^\s*$', "", regex=True)
        
        # Drop any completely empty rows
        df = df.dropna(how='all')
        
        # Get preview rows
        preview_df = df.head(max_rows)
        
        return {
            "filename": filename,
            "columns": list(df.columns),
            "preview_rows": preview_df.to_dict(orient="records"),
            "total_rows": len(df)
        }
        
    except pd.errors.ParserError as e:
        raise ValueError(f"Error parsing CSV data: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing file {filename}: {str(e)}")
