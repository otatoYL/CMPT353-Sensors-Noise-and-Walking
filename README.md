This project uses a smartphone's accelerometer sensor to analyze gait characteristics. By collecting and processing raw sensor data, we can extract information such as walking speed, step frequency, and analyze gait changes under different conditions.

Functions
- Data collection: collect acceleration data using cell phone sensors
- Data preprocessing: Remove noise and sensor drift
- Gait analysis: Recognize steps, calculate step frequency and walking speed
- Visualization: Graph acceleration and gait characteristics.

Use method
1. Install the dependencies: `pip install numpy pandas scipy matplotlib`. 
2. Collect data: Record acceleration data using an application such as Physics Toolbox Sensor Suite.
3. Run the analysis script: `python gait_analysis.py your_data_file.csv

File instruction
- `gait_analysis.py`: The main analysis script.
- `data/`: Stores collected sensor data.
- `results/`: Houses the analysis results and graphs.


Requirements
Python 3.6+
Required libraries: numpy, pandas, scipy, matplotlib
