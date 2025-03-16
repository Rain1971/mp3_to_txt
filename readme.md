# MP3 to Text Converter

This project converts MP3 audio files into text format. Follow the instructions below to set up the project and install the necessary dependencies.

## Prerequisites

Before running the project, ensure you have the following installed on your Windows machine:

1. **Python**: Make sure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/downloads/).

2. **FFmpeg**: This project requires FFmpeg for audio processing. Follow these steps to install FFmpeg on Windows:

   - Download the FFmpeg executable from the [FFmpeg official website](https://ffmpeg.org/download.html).
   - Extract the downloaded ZIP file to a folder (e.g., `C:\ffmpeg`).
   - Add the `bin` directory to your system's PATH:
     - Right-click on 'This PC' or 'My Computer' and select 'Properties'.
     - Click on 'Advanced system settings'.
     - Click on 'Environment Variables'.
     - In the 'System variables' section, find the `Path` variable and select it, then click 'Edit'.
     - Click 'New' and add the path to the `bin` directory (e.g., `C:\ffmpeg\bin`).
     - Click 'OK' to close all dialog boxes.

## Required Libraries

To install the necessary Python libraries, run the following command in your terminal:

```
pip install -r requirements.txt
```

## Usage

After setting up the environment and installing the required libraries, you can run the main script:

```
python src/main.py
```

Make sure to provide the path to your MP3 file when prompted.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.