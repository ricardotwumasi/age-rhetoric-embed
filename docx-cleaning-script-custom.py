import docx
import re

def clean_docx_document(doc):
    cleaned_text = []
    
    for para in doc.paragraphs:
        text = para.text.strip()
        
        # Skip empty paragraphs
        if not text:
            continue
        
        # Remove timestamps
        text = re.sub(r'^\d{2}:\d{2}:\d{2}\s+', '', text)
        
        # Extract speaker and content
        match = re.match(r'^(S\d+)\s+(.+)$', text)
        if match:
            speaker, content = match.groups()
            
            # Skip responses from S2 and S3
            if speaker in ['S2', 'S3']:
                continue
            
            # Add content without speaker label
            cleaned_text.append(content)
        else:
            # If no speaker label, add the whole text
            cleaned_text.append(text)
    
    return "\n\n".join(cleaned_text)  # Use double newline to separate different speakers' responses

# Load the DOCX document
doc_path = '/mnt/data/UA2.docx'
doc = docx.Document(doc_path)

# Clean the DOCX document
cleaned_text = clean_docx_document(doc)

# Save the cleaned text to a .txt file
cleaned_txt_path = '/mnt/data/cleaned_UA2_no_timestamps_no_S2_S3_no_labels.txt'
with open(cleaned_txt_path, 'w', encoding='utf-8') as txt_file:
    txt_file.write(cleaned_text)

print(f"Cleaned text saved to: {cleaned_txt_path}")

# Let's start by reading the content of the file to remove all timecodes in the format **:**:**
file_path = '/mnt/data/all_union_running_text_only.txt'

# Reading the file
with open(file_path, 'r') as file:
    content = file.read()

# Now let's use a regular expression to remove all timecodes in the format **:**:**
import re

# Regular expression pattern to match timecodes like 02:54:57
pattern = r'\b\d{2}:\d{2}:\d{2}\b'

# Removing the timecodes
processed_content = re.sub(pattern, '', content)

# Writing the processed content back to a new file
output_path = '/mnt/data/processed_text_without_timecodes.txt'
with open(output_path, 'w') as output_file:
    output_file.write(processed_content)
