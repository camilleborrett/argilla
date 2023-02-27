# Docker needs to be installed on the computer:
# Run following command in terminal: docker run -d --name quickstart -p 6900:6900 argilla/argilla-quickstart:latest

# install datasets library with pip install datasets
import argilla as rg
import pandas as pd


# load dataset
# you can load an Excel file with one column that contains all the text snippets that need to be annotated.

#data = pd.read_excel("data_to_be_annotated.xlsx")
#data = pd.read_csv("data_to_be_annotated.csv)

# this is just to create dummy data
snippets = {"snippet": ["Example of text to be annotated number 1. This text talks about politics",
          "Example of text to be annotated number 1. This text talks about health",
          "Example of text to be annotated number 1. This text talks about economics"]
 }

data = pd.DataFrame(snippets)

# the column with the text snippets to be annotated must be called "text"
data = data.rename(columns={"snippet": "text"})

# this creates an Argilla dataset with the above pandas dataframe
dataset_rg = rg.DatasetForTextClassification.from_pandas(data)

# optional two lines of code - this is if you want to pre-add the labels for classification,
# meaning the annotator cannot add new labels during the session
"""
settings = rg.TextClassificationSettings(label_schema=["Politics", "Economics", "Health"])
rg.configure_dataset(name="swp-un-resolutions", settings=settings)
"""

# create and load the Argilla dataset to start annotation UI (you can give it any name in the name argument)
# ====> click on link to localhost that appears after this line of code is run to open UI
# ====> username is argilla and password is 1234
rg.log(dataset_rg, name="swp-un-resolutions-1")


# ------------------- THIS PART IS TO BE RUN AFTER SOME ANNOTATIONS HAVE BEEN MADE ---------------------------
# once the annotation is completed you can export the data back into a pandas dataframe/excel/csv
# to be used for fine-tuning your model as usual
annotated_data = rg.load("swp-un-resolutions-1")

# iterate over the records in the dataset
inputs, labels = [], []
for record in annotated_data:
    # we only want records with annotations
    if record.annotation:
        inputs.append(record.text)
        labels.append(record.annotation)

# convert to pandas dataframe
training_data = pd.DataFrame()
training_data["text"] = inputs
training_data["label"] = labels
training_data.to_excel("training_data.xlsx")