---
title: On Device Vs Cloud Llm Inference
emoji: 📉
colorFrom: indigo
colorTo: yellow
sdk: static
pinned: false
---

# Project: On Device Vs Cloud Llm Inference
This project aims to compare the performance and latency of running small language model (SLM) inference on local devices in the browser versus cloud-based solutions.
Furthermore, it investigates two different scheduling policies to send request to the cloud or process them locally based the expected latency.

## Project Structure
- `analyse/`: Contains Jupyter notebooks used for the analysis and policy design.
- `dataset/`: Datasets used for the experiments.
- `results/`: Results from experiments, including latency measurements and performance metrics from different devices.
- `src/`: The JS source code for the project, including the services to send request to the cloud or process on device.
- `index.html`: The main HTML file to run the experiments in the browser.
- `styles.css`: CSS styles for the HTML file.

## Getting Started
To run the experiments, open `index.html` in a web browser. Ensure that you have an API key for the OpenRouter service to run the cloud inference.
You can then download the models and run them in the browser by leveraging the transformers.js library.


## Dataset preparation
To prepare the dataset for the experiments, we follow these steps:
- Download the dataset from kaggle or huggingface
- Add row indexes to each entry in the dataset for easy reference
- Save the prepared dataset in the `dataset/` directory