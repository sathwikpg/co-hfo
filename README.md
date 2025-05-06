# Co-hfo

This repository contains Python scripts for analyzing and visualizing high-frequency oscillation (HFO) synchronization in brain activity data. It includes methods for data preprocessing, event analysis, and generating various visualizations such as brain maps and temporal correlation graphs.

## Features

- **Brain Map Visualization**: Plot and visualize brain activity using 3D coordinate data.
- **Event Matching**: Identify matching files for encoding and recall trials based on common parts in filenames.
- **Coincidence Detection**: Analyze temporal coincidences in event data across multiple channels.
- **Error Plots**: Generate plots with mean and standard error to compare HFO synchronization across trials.

## Code Overview

### Key Functions
- `plot_brain_map(coords, node_values, trial)`: Visualizes brain activity using 3D coordinates and node values.
- `load_matching_files(encode_directory, recall_directory)`: Loada the encoding word files corresponding only to the words that were recalled.
- `find_coincidences(data_array, time_window)`: Detects temporal coincidences in event data.
- `process_folder(args)`: Processes HFO data for a given folder and trial type.
- `generate_sig_trial(trial, folder_path)`: Generates and visualizes significant trends for a particular trial type.

### Main Trials
- **ENCODE**
- **RECALL**
- **DISTRACTOR**
- **COUNTDOWN**

Each trial type is analyzed separately, and the results are plotted for comparison.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sathwikpg/co-hfo.git
   cd co-hfo
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Note**: Ensure you have Python 3.8+ installed.

## Usage

To analyze data and generate visualizations for the trials:
1. Define the path to your dataset in the `path_to_local` variable.
2. Run the script for all trials:
   ```bash
   python main.py
   ```
3. The results, including visualizations, will be saved in the output directory.

## Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nilearn`
- `multiprocessing`
- `pickle`

## Data Requirements

- The data should be organized into trial-specific folders (e.g., `ENCODE`, `RECALL`, etc.).
- Each trial folder should contain event files in `.pkl` format.
- Ensure that the filenames follow a consistent pattern to enable matching (e.g., `sub-word_start.pkl`).

## Example Outputs

1. **Brain Map Visualization**
   ![Example Brain Map](example_images/brain_map.png)

2. **Time-Series Analysis**
   ![Example Time-Series Plot](example_images/time_series.png)

## Contribution

Contributions are welcome! If you'd like to improve the code or add new features:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a detailed explanation.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the research community for providing tools and datasets that enable innovative brain activity analysis.
