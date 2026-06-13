import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).parent.parent.parent.parent.resolve())

from third_party.data_science.utils import get_raw_data_root_dir
from ds_agent.competition_instances import ALL_COMPETITIONS_LIST


def test_submission_file() -> int:
    raw_data_dir = Path(get_raw_data_root_dir())
    for i, competition in enumerate(ALL_COMPETITIONS_LIST):
        comp_dir = raw_data_dir / competition.competition_id.value
        if not comp_dir.exists():
            print(f"{comp_dir} does not exist")
        submission_file = comp_dir / competition.sample_submission_filename
        if not submission_file.exists():
            print(f"{i} - {submission_file} does not exist")

    return 0


if __name__ == "__main__":
    test_submission_file()
