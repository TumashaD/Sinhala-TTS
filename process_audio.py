import pandas as pd
import soundfile as sf
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def resample_audio(audio, orig_sr, target_sr):
    """
    Simple resampling using numpy interpolation
    Compatible with numpy 2.x
    """
    if orig_sr == target_sr:
        return audio
    
    # Calculate the new length
    new_length = int(len(audio) * target_sr / orig_sr)
    
    # Create new time indices
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    # Interpolate
    resampled = np.interp(new_indices, old_indices, audio)
    return resampled

def apply_preemphasis(audio, coef=0.97):
    """Apply pre-emphasis filter"""
    return np.append(audio[0], audio[1:] - coef * audio[:-1])

def process_dataset(dataset_path, meta_file, output_meta="phonemized_processed.csv"):
    """
    Process dataset:
    1. Resample all audio to 22050 Hz
    2. Filter out files that are too short or too long
    3. Normalize audio levels
    4. Create cleaned metadata file
    """
    
    # Configuration
    TARGET_SR = 22050
    MIN_DURATION = 2.0  # seconds
    MAX_DURATION = 15.0  # seconds
    
    # Read metadata
    metadata_path = os.path.join(dataset_path, meta_file)
    
    # Try to read the metadata file
    try:
        df = pd.read_csv(metadata_path, sep='|', header=None, 
                        names=['filename', 'text', 'phonemes'])
    except:
        # If that fails, try without phonemes column
        df = pd.read_csv(metadata_path, sep='|', header=None)
    
    print("=" * 80)
    print("AUDIO DATASET PROCESSOR")
    print("=" * 80)
    print(f"\nDataset path: {dataset_path}")
    print(f"Input metadata: {meta_file}")
    print(f"Output metadata: {output_meta}")
    print(f"\nConfiguration:")
    print(f"  Target sample rate: {TARGET_SR} Hz")
    print(f"  Duration range: {MIN_DURATION}-{MAX_DURATION} seconds")
    print(f"  Original dataset: {len(df)} files")
    print("=" * 80)
    
    # Create processed audio directory
    wavs_dir = os.path.join(dataset_path, "wavs")
    processed_dir = os.path.join(dataset_path, "wavs_processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    clean_data = []
    stats = {
        'too_short': 0,
        'too_long': 0,
        'resampled_44100': 0,
        'resampled_48000': 0,
        'resampled_other': 0,
        'already_22050': 0,
        'errors': 0,
        'missing': 0
    }
    
    # Store duration info for analysis
    original_durations = []
    processed_durations = []
    skipped_files = []
    
    print("\nProcessing audio files...\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Progress"):
        try:
            # Parse row - adjust based on your CSV format
            filename = str(row[0]) if not str(row[0]).endswith('.wav') else str(row[0])[:-4]
            text = str(row[1]) if len(row) > 1 else ""
            phonemes = str(row[2]) if len(row) > 2 else text
            
            # Original audio path
            audio_file = f"{filename}.wav"
            audio_path = os.path.join(wavs_dir, audio_file)
            
            if not os.path.exists(audio_path):
                stats['missing'] += 1
                skipped_files.append((audio_file, "missing"))
                continue
            
            # Load audio with soundfile
            audio, sr = sf.read(audio_path)
            
            # Handle stereo audio - convert to mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            duration = len(audio) / sr
            original_durations.append(duration)
            
            # Filter by duration
            if duration < MIN_DURATION:
                stats['too_short'] += 1
                skipped_files.append((audio_file, f"too_short_{duration:.2f}s"))
                continue
                
            if duration > MAX_DURATION:
                stats['too_long'] += 1
                skipped_files.append((audio_file, f"too_long_{duration:.2f}s"))
                continue
            
            # Resample to target sample rate
            if sr != TARGET_SR:
                audio = resample_audio(audio, sr, TARGET_SR)
                if sr == 44100:
                    stats['resampled_44100'] += 1
                elif sr == 48000:
                    stats['resampled_48000'] += 1
                else:
                    stats['resampled_other'] += 1
            else:
                stats['already_22050'] += 1
            
            # Normalize audio to prevent clipping
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95
            
            # Calculate processed duration
            processed_duration = len(audio) / TARGET_SR
            processed_durations.append(processed_duration)
            
            # Save processed audio
            new_audio_path = os.path.join(processed_dir, audio_file)
            sf.write(new_audio_path, audio, TARGET_SR)
            
            # Update path in metadata
            new_relative_path = f"{filename}"
            
            # Preserve the original format (with or without phonemes)
            if len(row) > 2:
                clean_data.append([new_relative_path, text, phonemes])
            else:
                clean_data.append([new_relative_path, text])
            
        except Exception as e:
            stats['errors'] += 1
            skipped_files.append((audio_file if 'audio_file' in locals() else f"row_{idx}", f"error: {str(e)}"))
            continue
    
    # Save cleaned metadata
    clean_df = pd.DataFrame(clean_data)
    output_path = os.path.join(dataset_path, output_meta)
    clean_df.to_csv(output_path, index=False, header=False, sep='|')
    
    # Save skipped files report
    if skipped_files:
        skipped_df = pd.DataFrame(skipped_files, columns=['filename', 'reason'])
        skipped_path = os.path.join(dataset_path, 'skipped_files.csv')
        skipped_df.to_csv(skipped_path, index=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\n📊 SUMMARY:")
    print(f"  Original files:        {len(df)}")
    print(f"  Processed files:       {len(clean_data)}")
    print(f"  Files removed:         {len(df) - len(clean_data)}")
    print(f"  Success rate:          {len(clean_data)/len(df)*100:.1f}%")
    
    print(f"\n❌ REMOVED FILES BREAKDOWN:")
    print(f"  Too short (<{MIN_DURATION}s):  {stats['too_short']} files")
    print(f"  Too long (>{MAX_DURATION}s):   {stats['too_long']} files")
    print(f"  Missing files:         {stats['missing']} files")
    print(f"  Errors:                {stats['errors']} files")
    
    print(f"\n🔄 RESAMPLING BREAKDOWN:")
    print(f"  From 44100 Hz:         {stats['resampled_44100']} files")
    print(f"  From 48000 Hz:         {stats['resampled_48000']} files")
    print(f"  From other rates:      {stats['resampled_other']} files")
    print(f"  Already 22050 Hz:      {stats['already_22050']} files")
    
    if processed_durations:
        processed_durations = np.array(processed_durations)
        print(f"\n⏱️  PROCESSED AUDIO STATISTICS:")
        print(f"  Mean duration:         {np.mean(processed_durations):.2f}s")
        print(f"  Median duration:       {np.median(processed_durations):.2f}s")
        print(f"  Min duration:          {np.min(processed_durations):.2f}s")
        print(f"  Max duration:          {np.max(processed_durations):.2f}s")
        print(f"  Total duration:        {np.sum(processed_durations)/3600:.2f} hours")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  Processed metadata:    {output_path}")
    print(f"  Processed audio:       {processed_dir}/")
    if skipped_files:
        print(f"  Skipped files report:  {skipped_path}")
    
    # Create visualization
    if processed_durations:
        plt.figure(figsize=(14, 5))
        
        # Duration histogram
        plt.subplot(1, 3, 1)
        plt.hist(processed_durations, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Duration (seconds)', fontsize=10)
        plt.ylabel('Number of files', fontsize=10)
        plt.title('Processed Audio Duration Distribution', fontsize=11, fontweight='bold')
        plt.axvline(MIN_DURATION, color='red', linestyle='--', label=f'Min: {MIN_DURATION}s')
        plt.axvline(MAX_DURATION, color='red', linestyle='--', label=f'Max: {MAX_DURATION}s')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 3, 2)
        plt.boxplot(processed_durations, vert=True)
        plt.ylabel('Duration (seconds)', fontsize=10)
        plt.title('Duration Box Plot', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Duration categories
        plt.subplot(1, 3, 3)
        categories = ['2-4s', '4-6s', '6-8s', '8-10s']
        counts = [
            np.sum((processed_durations >= 2) & (processed_durations < 4)),
            np.sum((processed_durations >= 4) & (processed_durations < 6)),
            np.sum((processed_durations >= 6) & (processed_durations < 8)),
            np.sum((processed_durations >= 8) & (processed_durations <= 10))
        ]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        plt.bar(categories, counts, color=colors, edgecolor='black', alpha=0.7)
        plt.xlabel('Duration Range', fontsize=10)
        plt.ylabel('Number of files', fontsize=10)
        plt.title('Files by Duration Category', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for i, count in enumerate(counts):
            plt.text(i, count + max(counts)*0.02, str(count), 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        viz_path = os.path.join(dataset_path, 'processed_audio_analysis.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization:         {viz_path}")
    
    print("\n" + "=" * 80)
    print("✅ Dataset processing complete!")
    print(f"✅ Update your config.json:")
    print(f'   "meta_file_train": "{output_meta}"')
    print("=" * 80 + "\n")
    
    return clean_df

if __name__ == "__main__":
    # Run the processor
    clean_df = process_dataset(
        dataset_path="dataset_cse",
        meta_file="phonemized.csv",  # Change to "phonemized.csv" if that's your file
        output_meta="phonemized_processed.csv"
    )