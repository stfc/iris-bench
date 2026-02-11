import pytest
import sys
from unittest.mock import patch, MagicMock
import argparse
from argparse import Namespace
from iris_gpubench.utils.cli import parse_arguments # Assuming iris_gpubench.utils.cli.py is in the current directory or importable

# Mock global constants and dependencies
MONITOR_INTERVAL = 5 # Defined in globals.py in the actual code
VALID_REGIONS = ["South England", "North Scotland", "Wales"]

# --- Fixtures and Mocks ---

@pytest.fixture
def mock_dependencies():
    """Fixture to mock LOGGER, image_exists, and list_available_images."""
    with (
        patch('iris_gpubench.utils.cli.LOGGER', new_callable=MagicMock) as mock_logger,
        patch('iris_gpubench.utils.cli.image_exists', new_callable=MagicMock) as mock_image_exists,
        patch('iris_gpubench.utils.cli.list_available_images', new_callable=MagicMock) as mock_list_images,
    ):
        yield mock_logger, mock_image_exists, mock_list_images

@pytest.fixture
def mock_carbon_regions():
    """Fixture for the get_carbon_region_names_func."""
    return MagicMock(return_value=VALID_REGIONS)

@pytest.fixture
def mock_parse_args_exit():
    """Fixture to mock parser.parse_args() and parser.error() to raise SystemExit."""
    # Patch sys.exit to raise SystemExit for all tests that expect an exit
    with patch('sys.exit', side_effect=SystemExit) as mock_exit:
        yield mock_exit


# --- Test Cases for Successful Parsing ---

def test_parse_arguments_benchmark_image_success(mock_dependencies, mock_carbon_regions):
    """Test successful parsing with --benchmark_image and default values."""
    mock_logger, mock_image_exists, mock_list_images = mock_dependencies
    mock_image_exists.return_value = True

    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_image', 'my/image']
    with patch('sys.argv', test_args):
        args = parse_arguments(mock_carbon_regions)

    assert isinstance(args, Namespace)
    assert args.benchmark_image == 'my/image'
    assert args.benchmark_command is None
    assert args.interval == MONITOR_INTERVAL
    assert args.carbon_region == 'South England'
    assert args.no_live_monitor is False
    assert args.no_plot is False
    assert args.live_plot is False
    assert args.export_to_meerkat is False
    assert args.monitor_logs is False
    assert args.nvidia_nsights is False
    mock_image_exists.assert_called_once_with('my/image')


def test_parse_arguments_benchmark_command_success(mock_dependencies, mock_carbon_regions):
    """Test successful parsing with --benchmark_command and default values."""
    mock_logger, mock_image_exists, mock_list_images = mock_dependencies
    mock_image_exists.return_value = True

    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_command', 'echo hello']
    with patch('sys.argv', test_args):
        args = parse_arguments(mock_carbon_regions)

    assert isinstance(args, Namespace)
    assert args.benchmark_command == 'echo hello'
    assert args.benchmark_image is None
    assert args.carbon_region == 'South England'
    mock_image_exists.assert_not_called()


def test_parse_arguments_all_options_success(mock_dependencies, mock_carbon_regions):
    """Test successful parsing with all flags and custom values."""
    mock_logger, mock_image_exists, mock_list_images = mock_dependencies
    mock_image_exists.return_value = True

    test_args = [
        'iris_gpubench.utils.cli.py',
        '--benchmark_image', 'other/image:latest',
        '--interval', '10',
        '--carbon_region', 'North Scotland',
        '--no_live_monitor',
        '--no_plot',
        '--live_plot',
        '--export_to_meerkat',
        '--monitor_logs',
        '--nvidia_nsights'
    ]
    with patch('sys.argv', test_args):
        args = parse_arguments(mock_carbon_regions)

    assert args.benchmark_image == 'other/image:latest'
    assert args.interval == 10
    assert args.carbon_region == 'North Scotland'
    assert args.no_live_monitor is True
    assert args.no_plot is True
    assert args.live_plot is True # Note: live_plot and no_plot are not mutually exclusive in the CLI logic
    assert args.export_to_meerkat is True
    assert args.monitor_logs is True
    assert args.nvidia_nsights is True
    mock_carbon_regions.assert_called_once()
    mock_image_exists.assert_called_once_with('other/image:latest')


# --- Test Cases for Argument Validation Failure ---


@patch.object(argparse.ArgumentParser, 'error', side_effect=SystemExit)
def test_parse_arguments_both_benchmark_fails(mock_error, mock_dependencies, mock_carbon_regions):
    """Test validation fails when both --benchmark_image and --benchmark_command are provided."""
    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_image', 'img', '--benchmark_command', 'cmd']
    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit):
            parse_arguments(mock_carbon_regions)

    mock_error.assert_called_once()
    mock_dependencies[0].error.assert_called_once() # Check LOGGER was called


@pytest.mark.parametrize("invalid_interval", [0, -5])
def test_parse_arguments_invalid_interval_fails(mock_dependencies, mock_carbon_regions, mock_parse_args_exit, invalid_interval):
    """Test validation fails when --interval is not a positive integer."""
    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_command', 'cmd', '--interval', str(invalid_interval)]

    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_arguments(mock_carbon_regions)

    assert excinfo.type is SystemExit
    mock_dependencies[0].error.assert_called_once() # Check LOGGER was called
    mock_parse_args_exit.assert_called_once_with(1)


def test_parse_arguments_invalid_carbon_region_fails(mock_dependencies, mock_carbon_regions, mock_parse_args_exit):
    """Test validation fails when --carbon_region is invalid."""
    mock_carbon_regions.return_value = VALID_REGIONS
    invalid_region = "Mars"
    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_command', 'cmd', '--carbon_region', invalid_region]

    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_arguments(mock_carbon_regions)

    assert excinfo.type is SystemExit
    mock_dependencies[0].error.assert_called_once() # Check LOGGER was called
    mock_parse_args_exit.assert_called_once_with(1)


def test_parse_arguments_non_existent_image_fails(mock_dependencies, mock_carbon_regions, mock_parse_args_exit):
    """Test validation fails when --benchmark_image is provided but the image does not exist."""
    mock_logger, mock_image_exists, mock_list_images = mock_dependencies
    mock_image_exists.return_value = False
    mock_list_images.return_value = ["img1", "img2"]
    test_image = "nonexistent:latest"
    test_args = ['iris_gpubench.utils.cli.py', '--benchmark_image', test_image]

    with patch('sys.argv', test_args):
        with pytest.raises(SystemExit) as excinfo:
            parse_arguments(mock_carbon_regions)

    assert excinfo.type is SystemExit
    mock_image_exists.assert_called_once_with(test_image)
    mock_list_images.assert_called_once()
    mock_logger.error.assert_called_once_with("Image '%s' does not exist.", test_image)
    mock_parse_args_exit.assert_called_once_with(1)