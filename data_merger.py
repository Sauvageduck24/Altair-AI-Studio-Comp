"""
Data Merger Script for Forex Data
==================================
This script merges the outlier_flag and cluster columns from results.csv 
with the original EURUSD_15M.csv data to create an enhanced dataset.

Features:
- Loads raw EURUSD 15-minute data
- Extracts outlier_flag and cluster information from results.csv
- Merges data based on row index/position
- Creates enhanced dataset with original data + outlier analysis

Author: Data Processing Pipeline
Date: October 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class ForexDataMerger:
    """
    Merges outlier detection and clustering results with original forex data.
    """
    
    def __init__(self):
        """Initialize the data merger."""
        self.raw_data = None
        self.results_data = None
        self.merged_data = None
        
    def load_raw_data(self, raw_data_path):
        """
        Load raw EURUSD forex data.
        
        Args:
            raw_data_path (str): Path to the raw EURUSD CSV file
            
        Returns:
            pd.DataFrame: Raw forex data
        """
        print(f"Loading raw data from: {raw_data_path}")
        self.raw_data = pd.read_csv(raw_data_path)
        
        # Convert time column to datetime
        self.raw_data['time'] = pd.to_datetime(self.raw_data['time'])
        
        print(f"Raw data loaded successfully.")
        print(f"Shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        print(f"Date range: {self.raw_data['time'].min()} to {self.raw_data['time'].max()}")
        
        return self.raw_data
    
    def load_results_data(self, results_data_path):
        """
        Load results data with outlier flags and clusters.
        
        Args:
            results_data_path (str): Path to the results CSV file
            
        Returns:
            pd.DataFrame: Results data with outlier and cluster information
        """
        print(f"Loading results data from: {results_data_path}")
        
        # Read the results file (it's semicolon-separated and has quotes)
        self.results_data = pd.read_csv(results_data_path, sep=';', quotechar='"')
        
        print(f"Results data loaded successfully.")
        print(f"Shape: {self.results_data.shape}")
        print(f"Available columns: {list(self.results_data.columns)}")
        
        # Check if required columns exist
        required_columns = ['outlier_flag', 'cluster']
        missing_columns = [col for col in required_columns if col not in self.results_data.columns]
        
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        else:
            print("✓ All required columns found: outlier_flag, cluster")
            
        # Show unique values in key columns
        if 'outlier_flag' in self.results_data.columns:
            print(f"Unique outlier_flag values: {self.results_data['outlier_flag'].unique()}")
            print(f"Outlier distribution: {self.results_data['outlier_flag'].value_counts().to_dict()}")
        
        if 'cluster' in self.results_data.columns:
            print(f"Unique cluster values: {self.results_data['cluster'].unique()}")
            print(f"Cluster distribution: {self.results_data['cluster'].value_counts().to_dict()}")
        
        return self.results_data
    
    def merge_data(self, merge_method='index'):
        """
        Merge raw data with results data based on the specified method.
        
        Args:
            merge_method (str): Method to merge data ('index' or 'position')
                - 'index': Merge based on DataFrame index
                - 'position': Merge based on row position (default)
                
        Returns:
            pd.DataFrame: Merged dataset with raw data + outlier flags and clusters
        """
        if self.raw_data is None:
            raise ValueError("Raw data not loaded. Call load_raw_data() first.")
            
        if self.results_data is None:
            raise ValueError("Results data not loaded. Call load_results_data() first.")
        
        print(f"Merging data using method: {merge_method}")
        
        # Create a copy of raw data to work with
        merged_data = self.raw_data.copy()
        
        # Determine the number of rows to merge (minimum between both datasets)
        min_rows = min(len(self.raw_data), len(self.results_data))
        print(f"Raw data rows: {len(self.raw_data)}")
        print(f"Results data rows: {len(self.results_data)}")
        print(f"Will merge first {min_rows} rows")
        
        if merge_method == 'index':
            # Merge based on index alignment
            if 'outlier_flag' in self.results_data.columns:
                merged_data.loc[:min_rows-1, 'outlier_flag'] = self.results_data['outlier_flag'].iloc[:min_rows].values
            
            if 'cluster' in self.results_data.columns:
                merged_data.loc[:min_rows-1, 'cluster'] = self.results_data['cluster'].iloc[:min_rows].values
                
        else:  # position-based merge (default)
            # Add outlier_flag column if it exists in results
            if 'outlier_flag' in self.results_data.columns:
                # Initialize with NaN for all rows
                merged_data['outlier_flag'] = np.nan
                # Fill first min_rows with results data
                merged_data.loc[:min_rows-1, 'outlier_flag'] = self.results_data['outlier_flag'].iloc[:min_rows].values
            
            # Add cluster column if it exists in results
            if 'cluster' in self.results_data.columns:
                # Initialize with NaN for all rows
                merged_data['cluster'] = np.nan
                # Fill first min_rows with results data
                merged_data.loc[:min_rows-1, 'cluster'] = self.results_data['cluster'].iloc[:min_rows].values
        
        # Add additional useful columns from results if they exist
        additional_columns = ['score', 'id']
        for col in additional_columns:
            if col in self.results_data.columns:
                merged_data[col] = np.nan
                merged_data.loc[:min_rows-1, col] = self.results_data[col].iloc[:min_rows].values
                print(f"✓ Added column: {col}")
        
        self.merged_data = merged_data
        
        print(f"\nMerge completed successfully!")
        print(f"Final dataset shape: {merged_data.shape}")
        print(f"Final columns: {list(merged_data.columns)}")
        
        return merged_data
    
    def get_merge_summary(self):
        """
        Get a summary of the merged dataset.
        
        Returns:
            dict: Summary statistics of the merged dataset
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge_data() first.")
        
        summary = {
            'total_records': len(self.merged_data),
            'columns': list(self.merged_data.columns),
            'date_range': {
                'start': str(self.merged_data['time'].min()),
                'end': str(self.merged_data['time'].max())
            }
        }
        
        # Outlier flag summary
        if 'outlier_flag' in self.merged_data.columns:
            outlier_counts = self.merged_data['outlier_flag'].value_counts(dropna=False)
            summary['outlier_distribution'] = outlier_counts.to_dict()
            
            # Calculate percentages
            total_non_null = self.merged_data['outlier_flag'].notna().sum()
            if total_non_null > 0:
                outlier_pct = {}
                for flag, count in outlier_counts.items():
                    if pd.notna(flag):
                        outlier_pct[flag] = (count / total_non_null) * 100
                summary['outlier_percentages'] = outlier_pct
        
        # Cluster summary
        if 'cluster' in self.merged_data.columns:
            cluster_counts = self.merged_data['cluster'].value_counts(dropna=False)
            summary['cluster_distribution'] = cluster_counts.to_dict()
            
            # Calculate percentages for clusters
            total_non_null = self.merged_data['cluster'].notna().sum()
            if total_non_null > 0:
                cluster_pct = {}
                for cluster, count in cluster_counts.items():
                    if pd.notna(cluster):
                        cluster_pct[cluster] = (count / total_non_null) * 100
                summary['cluster_percentages'] = cluster_pct
        
        return summary
    
    def save_merged_dataset(self, output_path, include_missing_data=True):
        """
        Save the merged dataset to a CSV file.
        
        Args:
            output_path (str): Path where to save the merged dataset
            include_missing_data (bool): Whether to include rows with missing outlier/cluster data
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge_data() first.")
        
        print(f"Saving merged dataset to: {output_path}")
        
        # Prepare data for saving
        data_to_save = self.merged_data.copy()
        
        if not include_missing_data:
            # Only keep rows where we have outlier/cluster information
            if 'outlier_flag' in data_to_save.columns:
                data_to_save = data_to_save[data_to_save['outlier_flag'].notna()]
                print(f"Filtered to {len(data_to_save)} rows with outlier information")
        
        # Save to CSV
        data_to_save.to_csv(output_path, index=False)
        
        print(f"✓ Dataset saved successfully!")
        print(f"Records saved: {len(data_to_save):,}")
        print(f"Columns saved: {len(data_to_save.columns)}")
        
    def create_sample_analysis(self, n_samples=10):
        """
        Create a sample analysis showing the merged data structure.
        
        Args:
            n_samples (int): Number of sample rows to display
            
        Returns:
            pd.DataFrame: Sample of merged data
        """
        if self.merged_data is None:
            raise ValueError("No merged data available. Call merge_data() first.")
        
        print("=" * 80)
        print("SAMPLE ANALYSIS OF MERGED DATASET")
        print("=" * 80)
        
        # Show first few rows
        sample = self.merged_data.head(n_samples)
        print(f"\nFirst {n_samples} rows of merged dataset:")
        print(sample.to_string())
        
        # Show data types
        print(f"\nData types:")
        for col, dtype in self.merged_data.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Show missing data summary
        print(f"\nMissing data summary:")
        missing_summary = self.merged_data.isnull().sum()
        for col, missing_count in missing_summary.items():
            if missing_count > 0:
                pct_missing = (missing_count / len(self.merged_data)) * 100
                print(f"  {col}: {missing_count:,} missing ({pct_missing:.2f}%)")
        
        return sample


def main():
    """
    Main execution function for data merging.
    """
    print("=" * 80)
    print("FOREX DATA MERGER - COMBINING RAW DATA WITH ANALYSIS RESULTS")
    print("=" * 80)
    
    # Define file paths
    raw_data_path = r"data_raw/EURUSD_15M.csv"
    results_data_path = r"data_preprocessed/results.csv"
    output_path = r"data_preprocessed/enhanced_eurusd_dataset.csv"

    # Initialize merger
    merger = ForexDataMerger()
    
    try:
        # Load data
        raw_data = merger.load_raw_data(raw_data_path)
        results_data = merger.load_results_data(results_data_path)
        
        print("\n" + "=" * 80)
        print("MERGING DATA")
        print("=" * 80)
        
        # Merge data
        merged_data = merger.merge_data(merge_method='position')
        
        # Show summary
        summary = merger.get_merge_summary()
        print(f"\n" + "=" * 80)
        print("MERGE SUMMARY")
        print("=" * 80)
        
        print(f"Total records: {summary['total_records']:,}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        if 'outlier_distribution' in summary:
            print(f"\nOutlier distribution:")
            for flag, count in summary['outlier_distribution'].items():
                print(f"  {flag}: {count:,}")
                
        if 'outlier_percentages' in summary:
            print(f"\nOutlier percentages:")
            for flag, pct in summary['outlier_percentages'].items():
                print(f"  {flag}: {pct:.2f}%")
        
        if 'cluster_distribution' in summary:
            print(f"\nCluster distribution:")
            for cluster, count in summary['cluster_distribution'].items():
                print(f"  {cluster}: {count:,}")
                
        if 'cluster_percentages' in summary:
            print(f"\nCluster percentages:")
            for cluster, pct in summary['cluster_percentages'].items():
                print(f"  {cluster}: {pct:.2f}%")
        
        # Create sample analysis
        merger.create_sample_analysis(n_samples=5)
        
        # Save merged dataset
        print(f"\n" + "=" * 80)
        print("SAVING ENHANCED DATASET")
        print("=" * 80)
        
        merger.save_merged_dataset(output_path, include_missing_data=True)
        
        print(f"\n" + "=" * 80)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Enhanced dataset saved to: {output_path}")
        print(f"You now have the original EURUSD data combined with outlier flags and cluster information!")
        
        return merged_data
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()