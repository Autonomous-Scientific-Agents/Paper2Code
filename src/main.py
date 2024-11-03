# src/main.py
import argparse
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(description='Paper2Code - Transform academic papers into functional code')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Create the database parser
    db_parser = subparsers.add_parser('create-database', help='Create software database from repository data')
    db_parser.add_argument('--output', type=str, help='Output file path for the database')

    # Add parser for paper processing (placeholder for future functionality)
    paper_parser = subparsers.add_parser('process-paper', help='Process academic paper')
    paper_parser.add_argument('--input', type=str, required=True, help='Input paper file path')
    paper_parser.add_argument('--output', type=str, help='Output directory for generated code')

    return parser.parse_args()


def read_file_content(file_path: Path) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def handle_database_creation(output_file: str = None) -> None:
    """Handle the database creation command."""
    from utils.database_creator import create_software_database

    # Get the project root directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Define input and output files
    data_dir = project_root / "examples" / "data"
    pypi_file = data_dir / "pypi_20241103_154558.json"
    conda_file = data_dir / "conda-forge_20241103_154558.json"
    bioconductor_file = data_dir / "bioconductor_20241103_154558.json"

    # Use provided output file or default
    if not output_file:
        output_dir = script_dir / "outputs"
        output_file = str(output_dir / "software_database.json")

    # Verify that input files exist
    if not all(f.exists() for f in [pypi_file, conda_file, bioconductor_file]):
        print("\nInput files not found. Checking paths:")
        for file_path in [pypi_file, conda_file, bioconductor_file]:
            print(f"{'✓' if file_path.exists() else '✗'} {file_path}")
        sys.exit(1)

    # Read repository data
    print("Reading repository data...")
    pypi_data = read_file_content(pypi_file)
    conda_data = read_file_content(conda_file)
    bioconductor_data = read_file_content(bioconductor_file)

    if not any([pypi_data, conda_data, bioconductor_data]):
        print("Error: No repository data could be read.")
        sys.exit(1)

    print("Creating software database...")
    try:
        create_software_database(
            pypi_data=pypi_data,
            conda_data=conda_data,
            bioconductor_data=bioconductor_data,
            output_file=output_file
        )
        print("\nDatabase creation completed successfully!")
    except Exception as e:
        print(f"Error creating database: {e}")
        sys.exit(1)


def handle_paper_processing(input_file: str, output_dir: str = None) -> None:
    """Handle the paper processing command (placeholder)."""
    print(f"Paper processing functionality will be implemented here.")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")


def main():
    args = parse_arguments()

    if args.command == 'create-database':
        handle_database_creation(args.output)
    elif args.command == 'process-paper':
        handle_paper_processing(args.input, args.output)
    else:
        print("Please specify a command. Use --help for more information.")
        sys.exit(1)


if __name__ == "__main__":
    main()
