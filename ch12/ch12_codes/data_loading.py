"""
Data Loading and Preprocessing Module
Chapter 12: Data-Led Technology Roadmapping

This module provides functions for loading and preprocessing
publication and patent data from bibliometric databases.
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_publication_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess publication data from bibliometric databases.
    
    Parameters:
        file_path: Path to CSV or Excel file containing publication data
        
    Expected columns:
        - Title: Publication title
        - Abstract: Full text abstract
        - Year: Publication year
        - Authors: Author names
        - Keywords: Author/index keywords
        - DOI: Digital Object Identifier
        - Citations: Citation count
        
    Returns:
        Preprocessed DataFrame with cleaned and standardized data
    """
    # Load data based on file extension
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    else:
        raise ValueError('Unsupported file format. Use CSV or Excel.')
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['Title', 'Year'])
    
    # Convert year to datetime
    df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
    df = df.dropna(subset=['Year'])
    
    # Fill missing abstracts with title
    df['Abstract'] = df['Abstract'].fillna(df['Title'])
    
    # Remove duplicates based on title
    df = df.drop_duplicates(subset=['Title'], keep='first')
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} publications from {file_path}")
    return df


def load_patent_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess patent data from patent databases.
    
    Parameters:
        file_path: Path to CSV file containing patent data
        
    Expected columns:
        - Patent_ID: Unique identifier
        - Title: Patent title
        - Abstract: Patent abstract
        - IPC_Class: IPC classification codes
        - Filing_Date: Application filing date
        - Assignee: Patent owner/applicant
        
    Returns:
        Preprocessed DataFrame with parsed IPC codes and dates
    """
    # Load data
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Parse IPC classifications (extract main class)
    if 'IPC_Class' in df.columns:
        df['IPC_Main'] = df['IPC_Class'].str[:4]
    
    # Convert filing date to datetime and extract year
    df['Filing_Date'] = pd.to_datetime(df['Filing_Date'], errors='coerce')
    df['Filing_Year'] = df['Filing_Date'].dt.year
    
    # Drop invalid entries
    df = df.dropna(subset=['Patent_ID', 'Filing_Year'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} patents from {file_path}")
    return df


def combine_publication_patent_data(pub_df: pd.DataFrame, 
                                     pat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine publication and patent data for integrated analysis.
    
    Parameters:
        pub_df: Publication DataFrame
        pat_df: Patent DataFrame
        
    Returns:
        Combined DataFrame with source indicator
    """
    # Standardize publication data
    pub_standardized = pub_df[['Title', 'Abstract', 'Year']].copy()
    pub_standardized['Source'] = 'Publication'
    pub_standardized['Year_Int'] = pub_standardized['Year'].dt.year
    
    # Standardize patent data
    pat_standardized = pat_df[['Title', 'Abstract', 'Filing_Year']].copy()
    pat_standardized.columns = ['Title', 'Abstract', 'Year_Int']
    pat_standardized['Source'] = 'Patent'
    pat_standardized['Year'] = pd.to_datetime(pat_standardized['Year_Int'], format='%Y')
    
    # Combine
    combined = pd.concat([pub_standardized, pat_standardized], ignore_index=True)
    
    print(f"Combined dataset: {len(combined)} documents "
          f"({len(pub_df)} publications, {len(pat_df)} patents)")
    return combined


# Example usage
if __name__ == "__main__":
    # Test with sample data
    print("Data Loading Module - Chapter 12")
    print("Use load_publication_data() and load_patent_data() functions")
    print("See README.md for complete documentation")
