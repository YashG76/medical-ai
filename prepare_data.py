from datasets import load_dataset, DatasetDict
import re

print("Loading FLEURS Gujarati dataset...")
ds = load_dataset('google/fleurs', 'gu_in')

def clean_text(batch):
    text = batch['transcription']
    # Remove punctuation that confuses Whisper
    text = re.sub(r'[,।\.!?;:]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    batch['transcription'] = text
    return batch

print("Cleaning transcriptions...")
ds = ds.map(clean_text)

# Keep only what Whisper needs
def keep_relevant_columns(batch):
    return {
        'audio': batch['audio'],
        'sentence': batch['transcription']
    }

ds = ds.map(keep_relevant_columns, remove_columns=ds['train'].column_names)

print("Saving cleaned dataset...")
ds.save_to_disk('data/gujarati_clean')

print("Done!")
print(ds)
print("Sample sentence:", ds['train'][0]['sentence'])