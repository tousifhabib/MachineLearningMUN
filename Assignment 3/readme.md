# SVM Classification for sDAT and sNC
This project demonstrates the use of Support Vector Machine (SVM) classification for distinguishing between sDAT (subsyndromal Delirium After Transplantation) and sNC (subsyndromal No Cognitive impairment) cases based on FDG-PET data.

## Dependencies
1. Python 3.7+
2. NumPy
3. pandas
4. scikit-learn
5. Matplotlib

## Usage
Ensure that the required dependencies are installed in your Python environment.
Update the file paths for train_sDAT_file_path, train_sNC_file_path, test_sDAT_file_path, and test_sNC_file_path in the if __name__ == "__main__" block with the appropriate file paths for your data.
The script will output the performance metrics and confusion matrix for each SVM kernel (linear, polynomial, and RBF) based on the provided data.

## Results
The script will perform the following tasks for each kernel:

1. Tune the SVM hyperparameters using GridSearchCV.
2. Train the SVM model on the training dataset.
3. Predict the classes for the test dataset.
4. Print the classification report containing precision, recall, f1-score, and support.
5. Plot the performance curves and confusion matrix for visualization.
