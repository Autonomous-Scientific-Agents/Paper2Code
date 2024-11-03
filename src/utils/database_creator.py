import json
from pathlib import Path
from typing import Dict, List, Union


class SoftwareDatabaseCreator:
    def __init__(self):
        self.database: Dict[str, Dict[str, Union[List[str], List[str]]]] = {}

    def process_repository_data(self, data: List[Dict], repository_name: str) -> None:
        """Process software data from a repository and add it to the database."""
        for item in data:
            if not isinstance(item, dict):
                continue

            name = item.get('name')
            versions = item.get('versions')

            if not name:
                continue

            if name not in self.database:
                self.database[name] = {
                    'versions': set(),
                    'repositories': set()
                }

            if versions:
                if isinstance(versions, list):
                    self.database[name]['versions'].update(versions)
                elif isinstance(versions, str):
                    self.database[name]['versions'].add(versions)

            self.database[name]['repositories'].add(repository_name)

    def format_database(self) -> Dict[str, Dict[str, Union[List[str], List[str]]]]:
        """Convert sets to sorted lists for JSON serialization."""
        formatted_db = {}
        for name, info in self.database.items():
            formatted_db[name] = {
                'versions': sorted(list(info['versions'])) if info['versions'] else None,
                'repositories': sorted(list(info['repositories']))
            }
        return formatted_db

    def save_database(self, output_file: str = 'software_database.json') -> None:
        """Save the formatted database to a JSON file."""
        formatted_db = self.format_database()
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(formatted_db, f, indent=2, sort_keys=True)
        print(f"Database saved to {output_file}")

    def get_statistics(self) -> Dict[str, int]:
        """Get basic statistics about the database."""
        formatted_db = self.format_database()
        stats = {
            'total_packages': len(formatted_db),
            'packages_with_versions': len([pkg for pkg in formatted_db.values() if pkg['versions']]),
            'multi_repo_packages': len([pkg for pkg in formatted_db.values() if len(pkg['repositories']) > 1])
        }
        return stats

    @staticmethod
    def load_json_data(json_str: str) -> List[Dict]:
        """Load JSON data from a string."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return []


def create_software_database(pypi_data: str, conda_data: str, bioconductor_data: str,
                           output_file: str = 'software_database.json') -> None:
    """
    Create a software database from multiple repository data sources.

    Args:
        pypi_data: JSON string containing PyPI package data
        conda_data: JSON string containing Conda Forge package data
        bioconductor_data: JSON string containing Bioconductor package data
        output_file: Path to save the resulting database
    """
    # Initialize database creator
    db_creator = SoftwareDatabaseCreator()

    # Process each repository
    repositories_data = [
        (pypi_data, 'pypi'),
        (conda_data, 'conda-forge'),
        (bioconductor_data, 'bioconductor')
    ]

    for data_str, repo_name in repositories_data:
        data = db_creator.load_json_data(data_str)
        if data:
            db_creator.process_repository_data(data, repo_name)
        else:
            print(f"Warning: No data processed for {repo_name}")

    # Save the database
    db_creator.save_database(output_file)

    # Print statistics
    stats = db_creator.get_statistics()
    print("\nDatabase Statistics:")
    print(f"Total unique packages: {stats['total_packages']}")
    print(f"Packages with version information: {stats['packages_with_versions']}")
    print(f"Packages in multiple repositories: {stats['multi_repo_packages']}")

    # Print sample entries
    print("\nSample entries:")
    formatted_db = db_creator.format_database()
    for name in list(formatted_db.keys())[:3]:
        print(f"\n{name}:")
        print(json.dumps(formatted_db[name], indent=2))
