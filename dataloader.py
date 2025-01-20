import csv

def content_generator(csv_path, start_idx=0, length=None):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Skip the specified number of rows
        for _ in range(start_idx):
            next(reader)
        
        rows_processed = 0
        for row in reader:
            if length is not None and rows_processed >= length:
                break
            
            content = f"以下是屬於{row['department']}的病例\n{row['summary']}"
            yield content
            rows_processed += 1