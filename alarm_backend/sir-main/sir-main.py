import pandas as pd
from datetime import timedelta
from collections import deque
import os
import json

def analyze_alarm_flooding(file_path, header_row=8, count_threshold=10, time_window_minutes=10):
    """
    Analyzes a CSV file for alarm flooding events.

    An alarm flooding event is defined as a specific alarm source appearing
    a certain number of times or more within a defined time window.

    Args:
        file_path (str): The path to the CSV file.
        header_row (int): The 0-indexed row number of the header.
        count_threshold (int): The minimum number of alarms to trigger a flooding event.
        time_window_minutes (int): The time window in minutes for the analysis.

    Returns:
        pd.DataFrame: A DataFrame with details of all identified flooding events.
    """
    try:
        print(f"\nAttempting to read file: {file_path}")
        # Load the CSV file, skipping the top rows that are not part of the data
        df = pd.read_csv(file_path, skiprows=header_row)
        print(f"Successfully loaded file with {len(df)} rows")

        # Rename columns for easier access
        df.columns = [
            'Event Time', 'Location Tag', 'Source', 'Condition', 'Action', 
            'Priority', 'Description', 'Value', 'Units'
        ]

        # Convert 'Event Time' column to a datetime format for time-based calculations
        # The format is specified to handle milliseconds correctly.
        df['Event Time'] = pd.to_datetime(df['Event Time'], format='%d/%m/%Y %H:%M:%S.%f', errors='coerce')
        
        # Drop rows where datetime conversion failed
        df.dropna(subset=['Event Time'], inplace=True)
        
        # Sort the DataFrame by time to ensure a chronological sequence
        df.sort_values(by='Event Time', inplace=True)

        # List to store the results of the analysis
        flooding_events = []
        
        # Get a list of unique alarm sources to iterate through
        unique_sources = df['Source'].unique()
        
        print(f"Starting analysis for {len(unique_sources)} unique alarm sources in file: {os.path.basename(file_path)}...")

        # Loop through each unique source to find flooding events
        for source in unique_sources:
            # Filter the DataFrame for the current alarm source
            source_alarms = df[df['Source'] == source].reset_index(drop=True)

            # Use a deque to maintain a sliding window of events within the time window
            window = deque()
            
            # Iterate through the alarms for the current source
            for idx, alarm in source_alarms.iterrows():
                # Add the current alarm's timestamp to the sliding window
                window.append(alarm['Event Time'])

                # Remove any alarms from the left of the window that are outside the 10-minute range
                while window and window[-1] - window[0] > timedelta(minutes=time_window_minutes):
                    window.popleft()

                # If the number of alarms in the window meets the threshold, it's a flooding event
                if len(window) >= count_threshold:
                    # Record the event details
                    flooding_events.append({
                        'File Path': file_path, # Add the file path to the report
                        'Alarm Source': source,
                        'Flood Count': len(window),
                        'Start Time': window[0].strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'End Time': window[-1].strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'Location': alarm['Location Tag'],
                        'Setpoint Value': alarm['Value']
                    })
                    
                    # Clear the window to avoid counting the same flooding event multiple times
                    window.clear()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred processing file '{file_path}': {e}")
        return pd.DataFrame()

    # Create a DataFrame from the collected flooding events
    results_df = pd.DataFrame(flooding_events)
    
    return results_df

def process_directory(root_directory, header_row=8, count_threshold=10, time_window_minutes=10):
    """
    Walks through a directory and its sub-directories to find and analyze
    all CSV files for alarm flooding events.

    Args:
        root_directory (str): The root path to the directory to start the search.
        header_row (int): The 0-indexed row number of the header in each file.
        count_threshold (int): The minimum number of alarms to trigger a flooding event.
        time_window_minutes (int): The time window in minutes for the analysis.

    Returns:
        pd.DataFrame: A single, combined DataFrame with all flooding events.
    """
    all_results = []
    print(f"Searching for CSV files in '{root_directory}'...")
    
    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                print(f"Processing file: {file_path}")
                # Analyze the current file and append the results
                file_results = analyze_alarm_flooding(
                    file_path, header_row, count_threshold, time_window_minutes
                )
                if not file_results.empty:
                    all_results.append(file_results)

    # Concatenate all the individual DataFrames into a single one
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()


# --- Main script execution ---
if __name__ == "__main__":
    # Define the root directory to analyze
    # NOTE: Replace 'ALARM_DATA_DIR' with the actual path to your folder
    alarm_data_dir = 'PVC-I (Jan, Feb, Mar) EVENTS'
    print(f"Looking for files in directory: {os.path.abspath(alarm_data_dir)}")
    
    # Run the analysis on all files in the directory
    flooding_report = process_directory(alarm_data_dir)

    # Save the report to a JSON file for visualization
    if not flooding_report.empty:
        # Convert the DataFrame to a list of dictionaries (JSON format)
        flooding_report_json = flooding_report.to_dict(orient='records')
        
        output_file = 'flooding_data.json'
        with open(output_file, 'w') as f:
            json.dump(flooding_report_json, f, indent=4)
        print(f"\n--- Alarm Flooding Report ---")
        print(f"Total flooding events found across all files: {len(flooding_report)}")
        print(f"Report saved to '{output_file}' for visualization.")
    else:
        print("\nNo alarm flooding events were detected in the specified directory.")
