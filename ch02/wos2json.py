"""
Web of Science (WoS) Data Converter

This module provides tools for converting Web of Science plain text export files
(.txt) to structured JSON format, with built-in data quality assessment capabilities.

The converter supports batch processing of multiple WoS export files and includes
automatic handling of multi-line fields, various character encodings, and
comprehensive field mapping for bibliometric analysis.

Usage:
    Basic conversion:
        python wos2json.py
    
    Automatic mode (no user confirmation):
        python wos2json.py --auto
    
    Programmatic usage:
        from wos2json import convert_wos_to_json
        convert_wos_to_json(['file1.txt', 'file2.txt'], 'output.json')

For technology foresight applications, see accompanying documentation on
data quality assessment and bibliometric network analysis.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


class WoSParser:
    """
    Parser for Web of Science plain text format files.
    
    This class handles the conversion of WoS-formatted bibliographic data
    into structured Python dictionaries. It manages multi-line fields,
    field code mapping, and maintains data quality tracking.
    
    Attributes:
        current_record (dict): Temporary storage for the record being parsed
        records (list): Collection of all parsed records
    """
    
    def __init__(self):
        """Initialize parser with empty record containers."""
        self.current_record = {}
        self.records = []
        
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse a single Web of Science export file.
        
        Handles multiple character encodings and reconstructs multi-line fields
        that are common in bibliographic data (e.g., abstracts, author lists).
        
        Args:
            file_path: Path to the WoS text file
            
        Returns:
            List of dictionaries, each representing one bibliographic record
            
        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        self.records = []
        self.current_record = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # Fallback to UTF-8 with BOM if standard UTF-8 fails
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # End of record marker
            if line == 'ER':
                if self.current_record:
                    self._extract_required_fields()
                    self.records.append(self.current_record)
                    self.current_record = {}
                i += 1
                continue
            
            # Skip empty lines
            if not line:
                i += 1
                continue
            
            # Parse field code and value (format: "XX value")
            if len(line) >= 2 and line[2:3] == ' ':
                field_code = line[:2]
                field_value = line[3:].strip()
                
                # Concatenate multi-line field content
                i += 1
                while i < len(lines):
                    next_line = lines[i].strip()
                    # Check for next field code or end of record
                    if (next_line and len(next_line) >= 2 and 
                        next_line[2:3] == ' ') or next_line == 'ER' or not next_line:
                        break
                    field_value += ' ' + next_line
                    i += 1
                
                # Store field in current record
                self._add_field(field_code, field_value)
                continue
            
            i += 1
        
        return self.records
    
    def _add_field(self, field_code: str, field_value: str):
        """
        Add a field to the current record with appropriate data structure.
        
        Handles both single-value and multi-value fields. If a field code
        appears multiple times, it converts to list structure.
        
        Args:
            field_code: Two-character WoS field tag (e.g., 'AU', 'TI')
            field_value: Content of the field
        """
        field_value = field_value.strip()
        
        if field_code in self.current_record:
            # Convert to list if field already exists
            if isinstance(self.current_record[field_code], list):
                self.current_record[field_code].append(field_value)
            else:
                self.current_record[field_code] = [
                    self.current_record[field_code], 
                    field_value
                ]
        else:
            self.current_record[field_code] = field_value
    
    def _extract_required_fields(self):
        """
        Map WoS field codes to standardized field names.
        
        Transforms raw WoS codes (e.g., TI, AU) into descriptive field names
        and structures multi-value fields as lists. Preserves original record
        for reference.
        
        Field Mapping:
            TI -> title
            SO -> journal_name
            SN -> issn_print
            EI -> issn_electronic
            PY -> publication_year
            AB -> abstract
            DI -> doi
            AU -> authors (list)
            AF -> authors_full (list)
            UT -> wos_id
            PM -> pubmed_id
            PT -> publication_type
            VL -> volume
            IS -> issue
            BP -> page_begin
            EP -> page_end
        """
        extracted = {}
        
        # Essential bibliographic fields
        extracted['title'] = self.current_record.get('TI', '')
        extracted['journal_name'] = self.current_record.get('SO', '')
        extracted['issn_print'] = self.current_record.get('SN', '')
        extracted['issn_electronic'] = self.current_record.get('EI', '')
        extracted['publication_year'] = self.current_record.get('PY', '')
        extracted['abstract'] = self.current_record.get('AB', '')
        extracted['doi'] = self.current_record.get('DI', '')
        
        # Author information
        authors = self.current_record.get('AU', '')
        if isinstance(authors, list):
            extracted['authors'] = authors
        elif authors:
            extracted['authors'] = [authors]
        else:
            extracted['authors'] = []
        
        # Full author names (when available)
        authors_full = self.current_record.get('AF', '')
        if isinstance(authors_full, list):
            extracted['authors_full'] = authors_full
        elif authors_full:
            extracted['authors_full'] = [authors_full]
        else:
            extracted['authors_full'] = []
        
        # Identifiers
        extracted['wos_id'] = self.current_record.get('UT', '')
        extracted['pubmed_id'] = self.current_record.get('PM', '')
        
        # Publication metadata
        extracted['publication_type'] = self.current_record.get('PT', '')
        extracted['volume'] = self.current_record.get('VL', '')
        extracted['issue'] = self.current_record.get('IS', '')
        extracted['page_begin'] = self.current_record.get('BP', '')
        extracted['page_end'] = self.current_record.get('EP', '')
        
        # Preserve original record for reference and validation
        extracted['_original_wos_record'] = self.current_record.copy()
        
        self.current_record = extracted


def convert_wos_to_json(input_files: List[str], 
                        output_file: str = None,
                        pretty_print: bool = True, 
                        include_original: bool = False):
    """
    Convert multiple WoS export files to consolidated JSON format.
    
    Processes a list of WoS plain text files and combines them into a single
    JSON output. Provides progress reporting and file statistics.
    
    Args:
        input_files: List of paths to WoS .txt files
        output_file: Path for output JSON file (default: 'wos_articles.json')
        pretty_print: Format JSON with indentation for readability
        include_original: Preserve original WoS field codes in output
    
    Returns:
        str: Path to the created JSON file
        
    Example:
        >>> files = ['export1.txt', 'export2.txt']
        >>> output = convert_wos_to_json(files, 'combined_data.json')
        Processing 2 files...
        ...
        >>> print(f"Created: {output}")
    """
    parser = WoSParser()
    all_records = []
    
    print(f"Processing {len(input_files)} file(s)...\n")
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
        
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            records = parser.parse_file(file_path)
            
            # Remove original records if not requested
            if not include_original:
                for record in records:
                    if '_original_wos_record' in record:
                        del record['_original_wos_record']
            
            all_records.extend(records)
            print(f"  -> Found {len(records)} records\n")
            
        except Exception as e:
            print(f"  -> Error: {str(e)}\n")
            continue
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = "wos_articles.json"
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        if pretty_print:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        else:
            json.dump(all_records, f, ensure_ascii=False)
    
    # Summary report
    print(f"\n{'=' * 60}")
    print("Conversion completed successfully!")
    print(f"Total records: {len(all_records)}")
    print(f"Output file: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    print(f"{'=' * 60}")
    
    return output_file


def process_directory(directory_path: str, 
                     output_file: str = None,
                     file_pattern: str = "*.txt"):
    """
    Process all WoS export files in a directory.
    
    Searches for files matching the specified pattern and converts them
    to a single JSON output. Useful for batch processing of multiple
    WoS data exports.
    
    Args:
        directory_path: Path to directory containing WoS files
        output_file: Path for output JSON file
        file_pattern: Glob pattern for file matching (default: '*.txt')
    
    Returns:
        str: Path to the created JSON file, or None if no files found
        
    Example:
        >>> process_directory('./wos_exports/', 'all_articles.json')
    """
    directory = Path(directory_path)
    
    # Find all matching files in directory
    input_files = list(directory.glob(file_pattern))
    input_files = [str(f) for f in input_files]
    
    if not input_files:
        print(f"Warning: No files matching '{file_pattern}' found in '{directory_path}'")
        return None
    
    return convert_wos_to_json(input_files, output_file)


def main(auto_mode=False):
    """
    Main execution function - processes all .txt files in script directory.
    
    Searches for WoS export files (.txt) in the same directory as the script
    and converts them to JSON format. Can run interactively (prompting user)
    or automatically (for batch processing).
    
    Args:
        auto_mode: If True, processes files without user confirmation
        
    Usage:
        Interactive mode:
            python wos2json.py
        
        Automatic mode:
            python wos2json.py --auto
    """
    
    print("=" * 60)
    print("WoS to JSON Converter")
    print("=" * 60)
    print()
    
    # Locate script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir:
        script_dir = os.getcwd()
    
    print(f"Directory: {script_dir}")
    print()
    
    # Find all .txt files in directory
    txt_files = []
    for file in os.listdir(script_dir):
        if file.endswith('.txt') and os.path.isfile(os.path.join(script_dir, file)):
            txt_files.append(os.path.join(script_dir, file))
    
    # Report findings
    print(f"Found {len(txt_files)} .txt file(s)")
    
    if len(txt_files) == 0:
        print("\nNo .txt files found in this directory!")
        print()
        print("Usage suggestions:")
        print("  1. Place your WoS export files (.txt) in the same directory as this script")
        print("  2. Run the script again")
        print("  3. Or use programmatically:")
        print()
        print("     from wos2json import convert_wos_to_json")
        print("     convert_wos_to_json(['file1.txt', 'file2.txt'], 'output.json')")
        return
    
    # List found files
    print()
    for i, file in enumerate(txt_files, 1):
        file_size = os.path.getsize(file) / 1024  # Size in KB
        print(f"   {i}. {os.path.basename(file)} ({file_size:.2f} KB)")
    
    print()
    print("-" * 60)
    
    # User confirmation (if not in auto mode)
    if not auto_mode:
        response = input(
            f"\nConvert {len(txt_files)} file(s) to JSON? (Y/N): "
        ).strip().upper()
        
        if response not in ['Y', 'YES']:
            print("Operation cancelled.")
            return
    else:
        print(f"\nAuto mode enabled - processing {len(txt_files)} file(s)...")
    
    print()
    print("Starting conversion...")
    print()
    
    # Create output filename
    output_file = os.path.join(script_dir, "wos_articles_all.json")
    
    # Execute conversion
    try:
        convert_wos_to_json(
            input_files=txt_files,
            output_file=output_file,
            pretty_print=True,
            include_original=False
        )
        
        print()
        print("Success! Conversion completed.")
        print(f"Output: {os.path.basename(output_file)}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


def auto_convert():
    """
    Automatic conversion without user prompts.
    
    Processes all .txt files in the script directory without asking for
    confirmation. Designed for integration into automated workflows and
    batch processing pipelines.
    
    Usage:
        Command line:
            python wos2json.py --auto
        
        Python code:
            from wos2json import auto_convert
            auto_convert()
    """
    main(auto_mode=True)


if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['--auto', '-a']:
        # Automatic mode
        auto_convert()
    else:
        # Interactive mode (prompts user for confirmation)
        main()
