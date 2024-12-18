#!/bin/bash

### Step 1: Clone repository

### Step 2: Create virtual environment
python3 -m venv .venv

### Step 3: Activate virtual environment
source .venv/bin/activate

### Step 4: Install dependencies
pip install -r requirements.txt

### Step 5: Unpack example data
unzip -o example_dataset.zip -d .