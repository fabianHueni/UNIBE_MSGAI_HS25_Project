# Modeling and Scaling of Generative AI Project
Project of the Modeling and Scaling of Generative AI Systems lecture at the University of Neuchatel.

## Overview
The project aims to transform images of analog medication lists (e.g., handwritten or printed lists) into structured digital formats. 
This involves several key steps:
- Image to Text Conversion: Utilizing a pre-trained docling model to extract text and tables from images.
- Mapping to Vocabulary: Converting the extracted text into a predefined vocabulary of medications. As a predefined vocabulary we use a csv-file with all FDA Drugs, available at https://www.kaggle.com/datasets/protobioengineering/united-states-fda-drugs-feb-2024. 
- Transform to Structured Format: Organizing the mapped data into a structured format such as JSON or CSV for further processing.

The project is oriented on the Granit Docling WebGPU demo on huggingface (https://huggingface.co/spaces/ibm-granite/granite-docling-258M-WebGPU).

## Run project
To run the project, follow these steps:
- Clone the repository to your local machine (git clone <repository_url>).
- Open the `index.html` file in a web browser to access the user interface.
- Download the pre-trained docling model in the UI of the application.
- Select an image to process.
- Click on the "Process" button to start the conversion.

## Requirements
The project depends on the following requirements, while all of them are included into the `index.html` file:
- The huggingface transformers.js library
- Tailwind CSS for styling
- A pre-trained docling model for image to text conversion from the huggingface hub.

## Structure
The project is structured as follows:
- `index.html`: The main HTML file containing the user interface.
- `styles.css`: The CSS file for styling the application.
- `index.js`: The JavaScript file containing the logic for the processing.
- `docling-html-parser.js`: A helper script for parsing the docling model output into html to render.
