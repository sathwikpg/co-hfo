# co-hfo: HFO Processing and Analysis Pipeline

This repository provides a comprehensive pipeline for processing, analyzing, and visualizing High-Frequency Oscillation (HFO) synchronization in brain activity data. It includes scripts for organizing data, splitting HFO detections into trial-specific segments, and generating various visualizations such as brain maps, time-series plots, and statistical analyses. The detections are based on the [epycom](https://github.com/ICRC-BME/epycom/) library.

## Features

- **Folder Creation**: Automatically generates folders based on HFO filenames.
- **HFO Event Processing**: Splits HFO data into trial-specific segments using event timing data.
- **Parallel Processing**: Leverages multiprocessing for faster execution.
- **Brain Map Visualization**: Plot and visualize brain activity using 3D coordinate data.
- **Event Matching**: Identify matching files for encoding and recall trials based on common parts in filenames.
- **Coincidence Detection**: Analyze temporal coincidences in event data across multiple channels.
- **Statistical Analysis**: Perform correlation-based analyses and calculate significant and non-significant trends.
- **ANOVA Testing**: Compute p-values for comparing different trial types using one-way ANOVA.
- **Error Plots**: Generate plots with mean and standard error to compare HFO synchronization across trials.

---

## Input Data

The pipeline requires:
1. **HFO Detections**: Post-processed HFO detections in `.pkl` format for the entire task. These detections should be generated using the [epycom](https://github.com/ICRC-BME/epycom/) library.
2. **Event Files**: Event timing data in `.pkl` format.

---

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

---

## Pipeline Workflow

### Step 1: Folder Creation
Run the script to create folders for each HFO file:
```bash
python create_folders.py --pkl_file /path/to/files_list.pkl --output_dir /path/to/output/folder
```

### Step 2: HFO Processing
Split HFO detections into trial-specific files:
```bash
python hfo_processing.py --hfo_dir /path/to/hfo_files \
                         --event_dir /path/to/event_files \
                         --output_dir /path/to/output \
                         --folders_pkl /path/to/folders_list.pkl \
                         --num_processes 10
```

### Step 3: Trial Analysis
Analyze processed trials to study HFO synchronization and generate visualizations:
```bash
python co_hfo_analysis.py --dataset /path/to/processed_trials \
                          --trials ENCODE RECALL DISTRACTOR COUNTDOWN \
                          --freq_range 60 500 \
                          --time_bin_size 10 \
                          --time_window 5
```

---

## Code Overview

### Key Functions
- `plot_brain_map(coords, node_values, trial)`: Visualizes brain activity using 3D coordinates and node values.
- `load_matching_files(encode_directory, recall_directory)`: Matches encoding and recall files based on filename patterns.
- `find_coincidences(data_array, time_window)`: Detects temporal coincidences in event data.
- `process_folder(args)`: Processes HFO data for a given folder and trial type.
- `generate_sig_trial(trial, folder_path)`: Generates and visualizes significant trends for a particular trial type.

### Main Trials
- **ENCODE**
- **RECALL**
- **DISTRACTOR**
- **COUNTDOWN**

Each trial type is analyzed separately, and the results are plotted for comparison.

---

## Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nilearn`
- `statsmodels`
- `pickle`
- `argparse`
- `mlxtend`

You can install these using pip:
```bash
pip install pandas numpy matplotlib seaborn nilearn statsmodels argparse mlxtend
```

---

## Example Outputs

1. **Brain Map Visualization**
   ![Example Brain Map](example_images/brain_map.png)

2. **Time-Series Analysis**
   ![Example Time-Series Plot](example_images/time_series.png)

3. **ANOVA p-values**
   ![Example p-values Plot](example_images/p_values.png)

---

## Output Overview

1. **Session-Specific Folders**:
   - Created during the folder creation step.

2. **Trial-Specific Files**:
   - `.pkl` files containing HFO data for individual trials.
   - Organized by trial type.

3. **Visualization Outputs**:
   - **Brain Maps**: Visualizations of HFO synchronization for different trials.
   - **Time-Series Plots**: Normalized mean and error plots comparing synchronization across trials.
   - **Statistical Plots**: ANOVA p-value plots for identifying significant differences.

---

## Source of HFO Detections

The HFO detections used in this pipeline are generated using the [epycom](https://github.com/ICRC-BME/epycom/) library. Please refer to their documentation for more details on how to obtain the detections.

---

## Contribution

Contributions are welcome! If you'd like to improve the code or add new features:
1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a detailed explanation.

---

## Acknowledgments

- [epycom](https://github.com/ICRC-BME/epycom) for HFO detection.
- Special thanks to the research community for providing tools and datasets that enable innovative brain activity analysis.

---

## License

This project is licensed under the [MIT License](LICENSE).
