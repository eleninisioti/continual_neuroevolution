#!/usr/bin/env python3
"""
Script to load CSV data from the data directory and display all features/columns.
"""

import os
import pandas as pd
from pathlib import Path

def main():
    # Get the script directory and construct path to data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return
    
    # Find all CSV files in the data directory
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}\n")
    
    # Process each CSV file
    for csv_file in csv_files:
        print("=" * 80)
        print(f"File: {csv_file.name}")
        print("=" * 80)
        
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file)
            
            # Display basic information
            print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"\nFeatures (columns):")
            print("-" * 80)
            
            # List all features with their index
            for idx, col in enumerate(df.columns, 1):
                print(f"{idx:3d}. {col}")
            
            print(f"\nTotal number of features: {len(df.columns)}")
            
            # Display data types summary
            print(f"\nData types summary:")
            print(df.dtypes.value_counts())
            
            # Display first few rows info
            print(f"\nFirst few rows preview:")
            print(df.head(3))
            
        except Exception as e:
            print(f"Error loading {csv_file.name}: {e}")
        
        print("\n")

if __name__ == "__main__":
    main()


