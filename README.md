# HFO Processing Pipeline

This repository contains scripts to process post-processed High-Frequency Oscillation (HFO) detections from sEEG data into trial-specific segments for further analysis. The detections are based on the [epycom](https://github.com/ICRC-BME/epycom/) library.

## Features

- **Folder Creation**: Automatically generates folders based on HFO filenames.
- **HFO Event Processing**: Splits HFO data into trial-specific segments using event timing data.
- **Parallel Processing**: Leverages multiprocessing for faster execution.
- **Modular Design**: Easy-to-reproduce and customizable scripts.

## Input Data

The pipeline requires:
1. **HFO Detections**: Post-processed HFO detections in `.pkl` format for the entire task. These detections should be generated using the [epycom](https://github.com/ICRC-BME/epycom/) library.
2. **Event Files**: Event timing data in `.pkl` format.

## Dependencies

Install the required Python libraries:
- `pandas`
- `numpy`
- `pickle`
- `matplotlib`
- `seaborn`
- `argparse`

You can install the dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

1. **Folder Creation**:
   Run the script to create folders for each HFO file:
   ```bash
   python create_folders.py --pkl_file /path/to/files_list.pkl --output_dir /path/to/output/folder
   ```

2. **HFO Processing**:
   Process sessions and split HFO detections into trials:
   ```bash
   python hfo_processing.py --hfo_dir /path/to/hfo_files \
                            --event_dir /path/to/event_files \
                            --output_dir /path/to/output \
                            --folders_pkl /path/to/folders_list.pkl \
                            --num_processes 10
   ```

   - `--hfo_dir`: Directory containing HFO `.pkl` files.
   - `--event_dir`: Directory containing event `.pkl` files.
   - `--output_dir`: Directory where processed trial data will be saved.
   - `--folders_pkl`: Path to the pickle file containing folder names.
   - `--num_processes`: Number of parallel processes to use (default is the number of CPU cores).

3. **Example Command**:
   ```bash
   python hfo_processing.py --hfo_dir /home/sathwik/hfo_data \
                            --event_dir /home/sathwik/event_data \
                            --output_dir /home/sathwik/processed_trials \
                            --folders_pkl /home/sathwik/folders_list.pkl \
                            --num_processes 4
   ```

## Output

The output consists of:
1. **Session-Specific Folders**: Created under the specified `--output_dir`.
2. **Trial-Specific Files**: HFO data split into `.pkl` files corresponding to individual trials. These files are saved in folders named after the trial type.

## Source of HFO Detections

The HFO detections used in this pipeline are generated using the [epycom](https://github.com/ICRC-BME/epycom/) library. Please refer to their documentation for more details on how to obtain the detections.

## Acknowledgments

- [epycom](https://github.com/ICRC-BME/epycom) for HFO detection.
- Special thanks to the research community for providing detection algorithms and tools for data analysis.

## License

This repository is licensed under the [MIT License](LICENSE).
