import csv
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# For Windows users, specify the path to the eSpeak NG DLL if needed
# EspeakWrapper.set_library('C:\\Program Files\\eSpeak NG\\libespeak-ng.dll')

# For macOS users, specify the path to the eSpeak library if needed
EspeakWrapper.set_library('/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.1.dylib')

def convert_file(input_path: str, output_path: str):
    rows_to_process = []
    count = 0
    
    # First pass: collect all rows that need processing
    with open(input_path, "r", encoding="utf-8", newline='') as f:
        reader = csv.reader(f, delimiter='|')
        
        for row in reader:
            if len(row) >= 4:
                file_id, _, sinhala, speaker = row[:4]
                if speaker.lower() == "mettananda":
                    count += 1
                    sinhala = sinhala.replace("\n", " ").replace("\r", " ").strip()
                    rows_to_process.append([file_id, sinhala])
    
    print(f"Found {count} rows to process. Starting batch phonemization...")
    
    # Extract all text for batch processing
    texts = [row[1] for row in rows_to_process]
    
    # Batch phonemize all texts at once
    phonemized_texts = phonemize(texts, language='si', strip=True, preserve_punctuation=True,punctuation_marks=';:,.!?¡¿—…"«»“”‘’\'"()[]{}=+-*/\\')
    
    # Combine results
    new_rows = []
    for i, (file_id, original_text) in enumerate(rows_to_process):
        ipa = phonemized_texts[i]
        new_rows.append([file_id, original_text, ipa])
        print(f"[{i+1}/{count}] Processed: {original_text[:50]}...")

    with open(output_path, "w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f, delimiter='|')
        writer.writerows(new_rows)

    print(f"[✓] Saved filtered metadata → {output_path}")

if __name__ == "__main__":
    # Example usage
    # convert_file("dataset/original.csv", "phonemized.csv")

    ph = phonemize("ආරම්භයේදී බංග්ලාදේශ කණ්ඩායමේ පිතිකරුවන් ශ්‍රී ලංකා පන්දු යවන්නන් හමුවේ දැඩි පීඩනයකට ලක්ව සිටි අතර, නුවන් තුෂාරගේ පළමු පන්දු වාරයේදී ම පළමු කඩුල්ල ලෙස ටන්සිඩ් හසන් දැවී ගියේ ය.", language='si', strip=True, preserve_punctuation=True,punctuation_marks=';:,.!?¡¿—…"«»“”‘’\'"()[]{}=+-*/\\')
    print(ph)