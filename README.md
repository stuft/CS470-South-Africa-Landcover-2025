# CS470-South-Africa-Landcover-2025
## The Code and data from SSU's CS470 FInal Project: Landcover identification in South Africa
<br />

### By: Shelby Anderson, Steven Tuft, Cooper Hochman

<br />
<br />
<div align="center">
  <a href="">
    <img src="Assets/screenshot01.png" alt="GEE" width="1000" height="400">
  </a>
</div>

<br />

<div>

### How to Run Code
#### Jupyter Notebooks
Our code is built in Jupyter Notebooks meant to run in Google Colab. The cells must be run in order, and will save the files to an attached Google Drive for use later. Some cells may take a while to run. Having a Colab Pro account to run it with A100s and High Ram is highly recommended for the fastest run time.
<br />
CS470CapstoneTest1(6) is the latest version of the code. 
<br />
Important Cells include:
<br />
- Load Data: this is where the GEE data and labels are pulled from your Google Drive. Make sure the listed directories point at those data's locations.
<br />
- Existing Prep: this takes multiple intermediary steps and combines them together to gather all the data and combine it and ready it for cleaning.
<br />
- Training and Testing Data filters: These use Mahalanobis distance to drop outlier pixels. There is a prominant config block for each of the two cells where it can be adjusted.
<br />
- Fit training data (SVM GridSearch): The main cell, where the model is built using gridsearch to find the best parameters for the data. Take a really long time to run, even with strong hardware.
<br />
- Fit testing data: Despite what is says, this just runs the training model on the testing data, giving the true accuacy at the end.
<br />
- Display confusion matrix: Does what it says on the tin. Show the CM with true numbers not percentages to show the weights of each as well.

#### Google Earth Engine
We also used Google Earth Engine to extract the Satelite Embeddings V1 data for our specific regions. The Data can be accessed with this link: https://code.earthengine.google.com/?accept_repo=users/tufts/Tests
<br />
It can also be found in the GEE folder in this repository.
<br />
GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL just maps the 2022 GEE embeddings over South Africa.
<br />
Test4 is the final version of the code used to extract the Satelite Embeddings V1 data.
<br />
ReMap maps both the cleaned pixel based data set as well as the original shapes, allowing a user to see which pixels were removed from those shapes.
<br />
In order to run these, run the files in GEE, and make sure the file paths at the top of each match your GEE storage paths.

</div>