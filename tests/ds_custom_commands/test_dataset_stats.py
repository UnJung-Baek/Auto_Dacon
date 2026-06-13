import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from agent.commands.ds_custom_commands import GetDatasetStatisticsCmd
from agent.memory import MemKey, Memory


@pytest.fixture
def mock_agent(tmp_path):
    # Mock agent and dependencies
    agent = Mock()
    agent.memory.retrieve.return_value = None
    agent.memory = Memory()

    agent.task.env.get_src_path.return_value = tmp_path
    return agent


@pytest.fixture
def sample_images(mock_agent):
    setup_dir = mock_agent.task.env.get_src_path()

    # Create two 2x2 RGB images
    img1 = np.zeros((2, 2, 3), dtype=np.uint8)
    img1[..., 1] = 64  # Green channel: 64
    img1[..., 2] = 128  # Blue channel: 128
    Image.fromarray(img1).save(setup_dir / "img1.png")

    img2 = np.zeros((2, 2, 3), dtype=np.uint8)
    img2[..., 0] = 64  # Red channel: 64
    img2[..., 1] = 128  # Green channel: 128
    img2[..., 2] = 192  # Blue channel: 192
    Image.fromarray(img2).save(setup_dir / "img2.png")

    # Create CSV with image paths
    df = pd.DataFrame({"image_path": ["img1.png", "img2.png"], "image2_path": ["img1.png", "img2.png"]})
    df = df.map(lambda f: setup_dir / f)
    df.to_csv(setup_dir / "train_img_input_map.csv", index=False)

    return setup_dir


def test_compare_stats(mock_agent, sample_images):
    # Run new implementation
    cmd = GetDatasetStatisticsCmd(max_workers=0)

    cmd.func(mock_agent, MemKey.IMG_DATA_STATISTICS)
    new_stats = mock_agent.memory.retrieve(MemKey.IMG_DATA_STATISTICS)

    # Compute expected values manually
    total_pixels = 2 * 2 * 2  # 2 images, 2x2 each
    sum_r = (0 * 4) + (64 * 4)  # 256
    sum_g = (64 * 4) + (128 * 4)  # 768
    sum_b = (128 * 4) + (192 * 4)  # 1280
    expected_mean = [sum_r / total_pixels, sum_g / total_pixels, sum_b / total_pixels]  # [32, 96, 160]
    expected_mean = [s / 255 for s in expected_mean]

    # Check new implementation's results
    assert new_stats["mean per channel"] == pytest.approx(expected_mean, abs=1e-3), (
        "New implementation has incorrect mean values"
    )

    # Run old implementation (simulated)
    # This is a simplified version of the old logic
    img1 = T.ToTensor()(Image.open(sample_images / "img1.png"))
    img2 = T.ToTensor()(Image.open(sample_images / "img2.png"))
    dataset = [img1, img2]
    sum_per_channel = torch.zeros(3)
    sum_sq_per_channel = torch.zeros(3)
    for img in dataset:
        sum_per_channel += img.sum(dim=(-2, -1))
        sum_sq_per_channel += (img**2).sum(dim=(-2, -1))

    mean_per_channel = sum_per_channel / total_pixels
    std_per_channel = (sum_sq_per_channel / total_pixels - mean_per_channel**2).sqrt()

    old_stats = {
        "mean per channel": mean_per_channel.tolist(),
        "standard deviation per channel": std_per_channel.tolist(),
    }

    # Compare old and new
    assert new_stats["mean per channel"] == pytest.approx(old_stats["mean per channel"], abs=1e-3), (
        "New implementation differs from old"
    )


@pytest.mark.skipif(
    lambda : not (Path(__file__).parent / "train_img_input_map.csv").exists(),
    reason=f"Add a testing file (train_img_input_map.csv) in {Path(__file__).parent}",
)
def test_stats_speed(mock_agent):
    # Run new implementation
    mock_agent.task.env.get_src_path.return_value = Path(__file__).parent

    start_time = time.time()

    cmd = GetDatasetStatisticsCmd(max_workers=0)
    cmd.func(mock_agent, MemKey.IMG_DATA_STATISTICS)
    st_stats = mock_agent.memory.retrieve(MemKey.IMG_DATA_STATISTICS)

    st_time = time.time()

    mock_agent.memory = Memory()
    cmd = GetDatasetStatisticsCmd(max_workers=32)
    cmd.func(mock_agent, MemKey.IMG_DATA_STATISTICS)
    mt_stats = mock_agent.memory.retrieve(MemKey.IMG_DATA_STATISTICS)

    st_duration = st_time - start_time
    mt_duration = time.time() - st_time
    assert st_duration > mt_duration, "Multithreaded is slower than single threaded!"
    print(f"Single threaded: {st_duration}, Multi-threaded: {mt_duration}")

    # Check new implementation's results
    assert st_stats == mt_stats, "Computed stats don't match"
