# Retail RAG Project Developer Documentation


## Setup

### Launching vLLM

To launch vLLM with the specified model, use the following command:
```
vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --max-model-len 4096
```


## Overview

This project consists of several Python scripts designed to process, clean, and analyze retail data. The main components are:

1. `cleanup.py`: Data cleaning and preprocessing
2. `match_sheets_by_row_id.py`: Merging Excel sheets
3. `eval.py`: Evaluation metrics and data analysis

## 1. cleanup.py

### Purpose
This script is responsible for cleaning and preprocessing the repair shop data, focusing on work order notes and brand/model information.

### Key Components

#### LLM Models
- Uses OpenAILLM with the "meta-llama/Llama-3.1-8B-Instruct" model
- Two instances: one for cleaning notes and another for extracting brand/model information

#### Data Cleaning Functions
- `preprocess_text()`: Removes lines starting with names and dates from work order notes
- `remove_brand_prefix()`: Removes specified prefixes from brand and model text
- `remove_brand_descriptors()`: Removes color names and non-essential terms from brand and model text
- `extract_brand_model_llm()`: Uses LLM to extract brand and model from cleaned text
- `process_note()`: Processes a single work order note
- `process_brand_model()`: Processes a single brand and model entry

#### Main Workflow
1. Loads data from a Parquet file
2. Filters rows based on work order note criteria
3. Applies cleaning functions to work order notes and brand/model information
4. Saves the cleaned data to a new Parquet file

## 2. match_sheets_by_row_id.py

### Purpose
This script merges two Excel sheets based on a common identifier.

### Key Components

#### Functions
- `process_chunk()`: Processes a chunk of data, merging it with the second sheet
- `convert_column_types()`: Converts column types in the merged DataFrame
- `merge_excel_sheets_to_parquet()`: Main function that orchestrates the merging process

#### Main Workflow
1. Loads both Excel sheets
2. Processes data in chunks using parallel processing
3. Merges chunks with the second sheet
4. Converts column types
5. Saves the merged data to a Parquet file

## 3. eval.py

### Purpose
This script contains imports for various evaluation metrics and data analysis tools.

### Key Components

#### Imported Libraries and Functions
- Numpy, Pandas, Matplotlib, Seaborn for data manipulation and visualization
- Scikit-learn metrics for model evaluation:
  - Mean Absolute Error
  - Median Absolute Error
  - Mean Squared Error
  - R-squared Score
  - Mean Absolute Percentage Error
  - Explained Variance Score
- Custom transformers:
  - FeatureCounter
  - CastToFloat32
  - FeatureEngineer
  - TextPreprocessor
- TfidfVectorizer for text feature extraction

## Usage

1. Data Cleaning:
   ```
   python cleanup.py
   ```

2. Merging Excel Sheets:
   ```
   python match_sheets_by_row_id.py
   ```

3. Evaluation and Analysis:
   Use the functions and metrics imported in `eval.py` as needed in your analysis scripts.

## Dependencies

- pandas
- numpy
- pyarrow
- scikit-learn
- matplotlib
- seaborn
- tqdm
- concurrent.futures

Ensure all dependencies are installed before running the scripts.

## Notes

- The LLM models require access to a running instance of the specified model.
- Adjust file paths and model configurations as needed for your environment.
- The `eval.py` script is primarily for importing evaluation metrics and should be used in conjunction with your specific analysis needs.

