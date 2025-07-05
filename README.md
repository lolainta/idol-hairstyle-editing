# Deep Learning Final Project

## Group Members
- Cheng-Liang Chi
- Zi-Hui Li
- Ting-Wan Chang

## Installation

The development environment is based on Python 3.8 and requires several dependencies. Follow these steps to set up the environment:

1. Clone the repository and its submodules:
   ```bash
   git clone --recurse-submodules <repository-url>
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the checkpoint of SAM2:
   ```bash
   cd sam2/checkpoints
   ./download_ckpts.sh
   ```

## Usage

To run the streamlit app, execute the following command:

```bash
streamlit run main.py
```

## Docker Deployment

To deploy the application using Docker, follow these steps:
1. Build the Docker image:
   ```bash
   docker run -p 8501:8501 --runtime=nvidia --gpus all -v ./data:/app/data tonychi/idol
   ```
