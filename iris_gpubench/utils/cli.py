"""
cli.py

This module contains the command-line interface (CLI) argument parsing logic for the GPU monitoring script.
It defines and configures the argument parser, validates the input arguments, and returns the parsed arguments.

The module is designed to be imported and used by the main GPU monitoring script.

Usage:
    from cli import parse_arguments
    args = parse_arguments()

Dependencies:
    - argparse: For creating and managing command-line arguments.
    - sys: For system-specific parameters and functions.
    - .carbon_metrics: For validating carbon region names.
    - .utils.globals: For accessing global constants.
    - .utils.docker_utils: For Docker-related utility functions.
"""

import argparse
import sys

from .docker_utils import image_exists, list_available_images
#from ..carbon_metrics import get_carbon_region_names
from .globals import LOGGER, MONITOR_INTERVAL


def parse_arguments(get_carbon_region_names_func):
    """
    Parses and validates command-line arguments for the GPU monitoring script.

    Returns:
        argparse.Namespace: Parsed and validated command-line arguments.

    Raises:
        SystemExit: If argument validation fails.
    """
    parser = argparse.ArgumentParser(
        description='Monitor GPU metrics and optionally export data to MeerkatDB.'
    )

    parser.add_argument('--no-live-monitor', action='store_true',
                        help='Disable live monitoring of GPU metrics (default is enabled).')

    parser.add_argument('--interval', type=int, default=MONITOR_INTERVAL,
                        help=f'Interval in seconds for collecting GPU metrics (default is {MONITOR_INTERVAL} seconds).')

    parser.add_argument('--carbon-region', type=str, default='South England',
                        help='Region shorthand for The National Grid ESO Regional Carbon Intensity API (default is "South England").')

    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting of GPU metrics (default is enabled).')

    parser.add_argument('--live-plot', action='store_true',
                        help='Enable live plotting of GPU metrics.')

    parser.add_argument('--export-to-meerkat', action='store_true',
                        help='Enable exporting of collected data to MeerkatDB.')

    parser.add_argument('--benchmark-image', type=str,
                        help='Docker container image to run as a benchmark.')

    parser.add_argument('--benchmark-command', type=str,
                        help='Command to run as a benchmark in a tmux session.')

    parser.add_argument('--monitor-logs', action='store_true',
                        help='Enable monitoring of container or tmux logs in addition to GPU metrics.')

    parser.add_argument('--nvidia-nsights', action='store_true',
                        help='Enable Nvidia Nsights to do extra GPU and CPU sampling.')

    args = parser.parse_args()

    # Validate arguments
    if not args.benchmark_image and not args.benchmark_command:
        LOGGER.error("Neither '--benchmark-image' nor '--benchmark-command' provided. One must be specified.")
        parser.error("You must specify either '--benchmark-image' or '--benchmark-command'.")

    if args.benchmark_image and args.benchmark_command:
        LOGGER.error("Both '--benchmark-image' and '--benchmark-command' provided. Only one must be specified.")
        parser.error("You must specify either '--benchmark-image' or '--benchmark-command', not both.")

    if args.interval <= 0:
        error_message = f"Monitoring interval must be a positive integer. Provided value: {args.interval}"
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)

    valid_regions = get_carbon_region_names_func()
    if args.carbon_region not in valid_regions:
        error_message = f"Invalid carbon region: {args.carbon_region}. Valid regions are: {', '.join(valid_regions)}"
        print(error_message)
        LOGGER.error(error_message)
        sys.exit(1)

    if args.benchmark_image and not image_exists(args.benchmark_image):
        print(f"Image '{args.benchmark_image}' is not valid.")
        LOGGER.error("Image '%s' does not exist.", args.benchmark_image)

        available_images = list_available_images(exclude_base=True, exclude_none=True)
        print("Available images (excluding 'base' images):")
        for image in available_images:
            print(f"  - {image}")
        sys.exit(1)

    return args
