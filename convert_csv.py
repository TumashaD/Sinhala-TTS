import csv
import sys
from g2p import convert_text

def process_csv_file(input_file, output_file):
    """
    Process CSV file where each line has format: ID|Sinhala_text|existing_ipa
    Convert Sinhala text using g2p and replace the existing IPA
    """
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        
        writer = csv.writer(outfile, delimiter='|')
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('|', 2)  # Split into max 3 parts
            if len(parts) != 3:
                print(f"Warning: Skipping malformed line {line_num}: {line}")
                continue
                
            record_id, sinhala_text, old_ipa = parts
            
            # Convert Sinhala text to IPA using your g2p script
            new_ipa = convert_text(sinhala_text)
            
            # Write the new row with updated IPA
            writer.writerow([record_id, sinhala_text, new_ipa])
            
            processed_count += 1
            if processed_count % 100 == 0:  # Show progress every 100 records
                print(f"Processed {processed_count} records...")
            
            # Show first few examples only
            if processed_count <= 2:
                print(f"Processed {record_id}:")
                print(f"  Original: {sinhala_text[:50]}{'...' if len(sinhala_text) > 50 else ''}")
                print(f"  Old IPA:  {old_ipa[:50]}{'...' if len(old_ipa) > 50 else ''}")
                print(f"  New IPA:  {new_ipa[:50]}{'...' if len(new_ipa) > 50 else ''}")
                print()
    
    return processed_count

if __name__ == "__main__":
    # Input and output file paths
    input_file = "dataset_pathnirvana/metadata.csv"
    output_file = "converted_metadata.csv"
    
    print("Converting Sinhala text to IPA using your g2p script...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print("=" * 80)
    
    try:
        processed_count = process_csv_file(input_file, output_file)
        print("=" * 80)
        print(f"Conversion completed! Processed {processed_count} records.")
        print(f"Output saved to: {output_file}")
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")