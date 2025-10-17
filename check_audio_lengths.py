import os
import soundfile as sf
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_audio_lengths(dataset_path):
    """
    Check audio file lengths in the dataset and provide statistics.
    """
    wavs_dir = os.path.join(dataset_path, 'wavs')
    metadata_file = os.path.join(dataset_path, 'metadata.csv')
    
    if not os.path.exists(wavs_dir):
        print(f"Error: Directory {wavs_dir} does not exist!")
        return
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file {metadata_file} does not exist!")
        return
    
    # Load metadata
    metadata = pd.read_csv(metadata_file, sep='|', header=None, 
                          names=['filename', 'text', 'phonemes'])
    
    print(f"Found {len(metadata)} entries in metadata.csv")
    
    # Get all audio files
    audio_files = []
    durations = []
    sample_rates = []
    missing_files = []
    
    print("\nAnalyzing audio files...")
    
    for idx, row in metadata.iterrows():
        filename = row['filename'] + '.wav'
        filepath = os.path.join(wavs_dir, filename)
        
        if os.path.exists(filepath):
            try:
                # Load audio and get duration using soundfile
                y, sr = sf.read(filepath)
                duration = len(y) / sr
                
                audio_files.append(filename)
                durations.append(duration)
                sample_rates.append(sr)
                
                if idx % 100 == 0:
                    print(f"Processed {idx + 1}/{len(metadata)} files...")
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                missing_files.append(filename)
        else:
            missing_files.append(filename)
    
    # Convert to numpy arrays for statistics
    durations = np.array(durations)
    sample_rates = np.array(sample_rates)
    
    print(f"\n{'='*60}")
    print("AUDIO DATASET ANALYSIS RESULTS")
    print(f"{'='*60}")
    
    print(f"\nDataset: {dataset_path}")
    print(f"Total files in metadata: {len(metadata)}")
    print(f"Successfully processed: {len(audio_files)}")
    print(f"Missing/corrupted files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMissing files:")
        for file in missing_files[:10]:  # Show first 10 missing files
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    if len(durations) > 0:
        print(f"\n{'='*30} AUDIO DURATION STATISTICS {'='*30}")
        print(f"Duration statistics (seconds):")
        print(f"  Mean duration:    {np.mean(durations):.2f} seconds")
        print(f"  Median duration:  {np.median(durations):.2f} seconds")
        print(f"  Min duration:     {np.min(durations):.2f} seconds")
        print(f"  Max duration:     {np.max(durations):.2f} seconds")
        print(f"  Std deviation:    {np.std(durations):.2f} seconds")
        
        print(f"\nDuration distribution:")
        print(f"  < 2 seconds:      {np.sum(durations < 2)} files ({np.sum(durations < 2)/len(durations)*100:.1f}%)")
        print(f"  2-5 seconds:      {np.sum((durations >= 2) & (durations < 5))} files ({np.sum((durations >= 2) & (durations < 5))/len(durations)*100:.1f}%)")
        print(f"  5-10 seconds:     {np.sum((durations >= 5) & (durations < 10))} files ({np.sum((durations >= 5) & (durations < 10))/len(durations)*100:.1f}%)")
        print(f"  > 10 seconds:     {np.sum(durations >= 10)} files ({np.sum(durations >= 10)/len(durations)*100:.1f}%)")
        
        print(f"\nTotal audio duration: {np.sum(durations)/3600:.2f} hours")
        
        print(f"\n{'='*30} SAMPLE RATE STATISTICS {'='*30}")
        unique_srs = np.unique(sample_rates)
        print(f"Sample rates found:")
        for sr in unique_srs:
            count = np.sum(sample_rates == sr)
            print(f"  {sr} Hz: {count} files ({count/len(sample_rates)*100:.1f}%)")
        
        # Find outliers (very short or very long files)
        print(f"\n{'='*30} OUTLIER ANALYSIS {'='*30}")
        
        # Very short files (< 1 second)
        short_files = np.where(durations < 1.0)[0]
        if len(short_files) > 0:
            print(f"\nVery short files (< 1 second): {len(short_files)}")
            for idx in short_files[:5]:  # Show first 5
                print(f"  {audio_files[idx]}: {durations[idx]:.2f}s")
            if len(short_files) > 5:
                print(f"  ... and {len(short_files) - 5} more")
        
        # Very long files (> 15 seconds)
        long_files = np.where(durations > 15.0)[0]
        if len(long_files) > 0:
            print(f"\nVery long files (> 15 seconds): {len(long_files)}")
            for idx in long_files[:5]:  # Show first 5
                print(f"  {audio_files[idx]}: {durations[idx]:.2f}s")
            if len(long_files) > 5:
                print(f"  ... and {len(long_files) - 5} more")
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(durations, bins=50, alpha=0.7, color='blue')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('Number of files')
        plt.title('Distribution of Audio Durations')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(durations)
        plt.ylabel('Duration (seconds)')
        plt.title('Audio Duration Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_path, 'audio_duration_analysis.png'), dpi=150, bbox_inches='tight')
        print(f"\nHistogram saved as: {os.path.join(dataset_path, 'audio_duration_analysis.png')}")
        
        # Save detailed results to CSV
        results_df = pd.DataFrame({
            'filename': audio_files,
            'duration_seconds': durations,
            'sample_rate': sample_rates
        })
        results_df = results_df.sort_values('duration_seconds')
        results_file = os.path.join(dataset_path, 'audio_length_analysis.csv')
        results_df.to_csv(results_file, index=False)
        print(f"Detailed results saved as: {results_file}")
        
    print(f"\n{'='*60}")

if __name__ == "__main__":
    dataset_path = "dataset_cse"
    check_audio_lengths(dataset_path)