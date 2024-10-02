import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import concurrent.futures

def process_chunk(chunk, sheet2_data):
    print(f"Processing chunk with {len(chunk)} rows")
    print(f"Chunk columns: {chunk.columns.tolist()}")
    print(f"Sheet2 columns: {sheet2_data.columns.tolist()}")
    merged = pd.merge(chunk, sheet2_data, left_on='WO#', right_on='ID', how='inner')
    print(f"Merged chunk has {len(merged)} rows")
    print(f"Merged columns: {merged.columns.tolist()}")
    if 'ID_y' in merged.columns:
        merged = merged.drop(columns=['ID_y'])
    return merged

def convert_column_types(df):
    # Define column types
    date_columns = ['Date In', 'Due On', 'Date', 'ID']
    quantity_columns = ['Qty']
    dollar_columns = []
    
    for col in df.columns:
        if col in date_columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif col in quantity_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        elif col in dollar_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        elif df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    return df

def merge_excel_sheets_to_parquet(input_file, output_file, num_threads=32, chunk_size=10000):
    print("Loading workbook...")
    
    # Read Sheet 2 into memory
    sheet2 = pd.read_excel(input_file, sheet_name=1)
    print(f"Sheet 2 has {len(sheet2)} rows")
    print(f"Sheet 2 columns: {sheet2.columns.tolist()}")
    
    print("Processing and merging data...")
    
    # Read the entire Sheet 1 to get the correct number of rows
    sheet1 = pd.read_excel(input_file, sheet_name=0)
    total_rows = len(sheet1)
    print(f"Sheet 1 has {total_rows} rows")
    print(f"Sheet 1 columns: {sheet1.columns.tolist()}")
    
    # Calculate the number of chunks
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"Processing in {num_chunks} chunks")
    
    # Process chunks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        
        for chunk_start in range(0, total_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_rows)
            chunk = sheet1.iloc[chunk_start:chunk_end]
            
            future = executor.submit(process_chunk, chunk, sheet2)
            futures.append(future)
        
        # Process results as they complete
        results = []
        total_rows_processed = 0
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_chunks, desc="Processing chunks"):
            try:
                result = future.result()
                results.append(result)
                total_rows_processed += len(result)
            except Exception as e:
                print(f"Error processing chunk: {e}")
    
    # Combine all results
    final_df = pd.concat(results, ignore_index=True)
    print(f"Total rows processed: {total_rows_processed}")
    print(f"Final dataframe shape: {final_df.shape}")
    
    # Convert column types
    final_df = convert_column_types(final_df)
    
    # Write to Parquet
    print(f"Writing to Parquet file: {output_file}")
    table = pa.Table.from_pandas(final_df)
    pq.write_table(table, output_file)
    
    print(f"Merged data saved to {output_file}")

# Usage
input_file = "data.xlsx"
output_file = "merged_output.xlsx"
merge_excel_sheets_to_parquet(input_file, output_file, num_threads=32, chunk_size=1000)
