import pandas as pd

def process_cat_data():
    input_file = "mtDNA_sequences.txt"
    output_file = "cat_data.txt"
    
    print(f"Processing {input_file}...")
    
    sequences = []
    current_seq = []
    
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            
        # Skip the first line if it's just text (e.g., "Mitochondrial DNA sequences...")
        start_index = 0
        for i, line in enumerate(lines):
            if line.startswith(">"):
                start_index = i
                break
        
        # Parse FASTA format
        for line in lines[start_index:]:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    # Save previous sequence (chunked into 200bp to match human data)
                    full_seq = "".join(current_seq)
                    # Break long genome into smaller pieces
                    n = 200
                    chunks = [full_seq[i:i+n] for i in range(0, len(full_seq), n)]
                    sequences.extend(chunks)
                    current_seq = []
            else:
                current_seq.append(line)
                
        # Save the last one
        if current_seq:
            full_seq = "".join(current_seq)
            n = 200
            chunks = [full_seq[i:i+n] for i in range(0, len(full_seq), n)]
            sequences.extend(chunks)

        # Create DataFrame
        # Filter out tiny chunks (less than 100bp)
        sequences = [s for s in sequences if len(s) >= 100]
        
        df = pd.DataFrame(sequences, columns=['sequence'])
        df['class'] = 2 # Label for Cat
        
        # Save
        df.to_csv(output_file, sep='\t', index=False)
        print(f"✅ Success! Created '{output_file}' with {len(df)} sequences.")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'. Make sure it's in the same folder.")

if __name__ == "__main__":
    process_cat_data()