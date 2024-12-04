# README for Advanced Driver Monitoring System

## Overview

The **Advanced Driver Monitoring System** is a Python-based application designed to monitor driver behavior in real-time using computer vision techniques. It utilizes various machine learning models to detect facial landmarks, eye blinks, yawns, and hand positions, providing alerts for potential drowsiness or distractions.

## Features

- **Real-time Monitoring**: Captures video from a camera and processes frames to detect driver behavior.
  
- **Facial Landmark Detection**: Uses MediaPipe to identify facial features and expressions.
  
- **Drowsiness Detection**: Monitors eye aspect ratio (EAR) to detect drowsiness based on blink patterns.
  
- **Yawning Detection**: Uses mouth aspect ratio (MAR) to identify yawning behavior.
  
- **Hand Position Tracking**: Detects if the driver's hand is near their face, indicating potential distraction.
  
- **Age Classification {WIP}**: Estimates the driver's age using a pre-trained model to ensure compliance with age-related driving regulations.

- **Alerts and Logging {WIP}**: Provides visual alerts on the screen and logs events for further analysis.

- **Machine Learning Model Training {FUTURE}**: Allows users to train their own models for improved accuracy.

- **User Interface {FUTURE}**: Offers a simple and intuitive interface for users to monitor and adjust settings.

- **Data Visualization {FUTURE}**: Provides insights into driver behavior through interactive visualizations.

- **Integration with Other Systems {FUTURE}**: Allows integration with other systems, such as vehicle control.

- **Security and Privacy {FUTURE}**: Ensures secure data storage and transmission.
## Requirements

To run this application, ensure you have the following installed:

- Python 3.12 or higher
- OpenCV
- Mediapipe
- NumPy
- PyTorch
- Transformers
- TensorFlow
- SciPy
- PIL (Pillow)

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe numpy torch transformers tensorflow scipy pillow
```

## Installation

1. Clone the repository or download the source code files.

2. Navigate to the project directory in your terminal.

3. Install the required libraries as mentioned above.

4. Ensure you have a compatible camera connected to your system.

## Usage

To start monitoring, run the following command in your terminal:

```bash
python driver_monitor.py
```

### Parameters

You can customize the monitoring settings by modifying parameters in the `DriverMonitor` class:

- **Camera Source**: Change the `source` parameter in `start_monitoring()` method if you want to use an external camera or a video file.
  
- **Thresholds**: Adjust thresholds for detecting drowsiness and yawning in `initialize_thresholds()` method.

### Example

Upon running the script, a window will open displaying the video feed from your camera. The system will overlay detection results such as:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Alerts for excessive blinking, yawning, or hand near face.

Press `q` to exit the monitoring window.

## Code Structure

The main components of the code are as follows:

- **Imports**: All necessary libraries are imported at the beginning of the script.
  
- **Class Definition**: The main functionality is encapsulated in the `DriverMonitor` class.
  
- **Initialization**: The constructor initializes models and metrics used for detection.
  
- **Frame Processing**: The `process_frame()` method handles frame-by-frame analysis including facial landmark detection, eye and mouth aspect ratio calculations, and alert generation.

## Logging

The system generates a log file (`driver_log_<timestamp>.txt`) that records alerts and metrics during operation. This file can be reviewed for further analysis of driver behavior over time.

## Troubleshooting

If you encounter issues:

- Ensure your camera is connected and accessible.
  
- Check that all required libraries are installed correctly.
  
- Review console output for error messages that can help diagnose problems.

## Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request.
