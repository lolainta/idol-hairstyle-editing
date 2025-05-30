# Deep Learning Final Project



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
