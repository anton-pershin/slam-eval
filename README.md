# slam-eval
Evaluation library for LLMs and other foundational models

## Getting started

1. Create a virtual environment, e.g.
```bash
conda create -n slam_eval python=3.12
conda activate slam_eval
```
2. Install necessary packages
```bash
pip install -r requirements.txt
```
3. Set up `/config/user_settings/user_settings.yaml`
4. Run one of the scripts `/project_name/scripts/main.py` and do not forget to modify the corresponding config file in `/config/config_main.yaml'
```bash
python project_name/scripts/main.py
```

⚠️  DO NOT commit your `user_settings.yaml`

## Scripts

### `main.py`

Runs evaluation

#### Configuration

1. In `user_settings.yaml`, set up your XXX:
   ```yaml
    ...
   ```

2. In `config_main.yaml`, modify:
   - ...

#### Output

Creates XXX
