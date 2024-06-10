# Gender Prediction using SVM Model

This project aims to predict gender based on facial images using Support Vector Machine (SVM) model. The dataset used in this project is taken from the UCI repository, Burden Diseases.

## Overview

The main objective of this project is to develop a machine learning model that can accurately predict the gender of individuals from facial images. Various techniques such as image augmentation, Principal Component Analysis (PCA), and baseline models have been explored to enhance the prediction accuracy.

## Dataset

The dataset used in this project is sourced from the UCI repository, Burden Diseases. It consists of a collection of facial images labeled with gender information.

## Techniques Explored

1. **Image Augmentation**: Augmentation techniques have been applied to increase the diversity and size of the dataset, enhancing the robustness of the model.

2. **Principal Component Analysis (PCA)**: PCA has been utilized for dimensionality reduction to improve model performance and efficiency.

3. **Baseline Model**: The most_frequent baseline model has been implemented as a benchmark for comparison. Despite its simplicity, it provides insights into the prediction accuracy achievable without sophisticated algorithms.

## Results

The SVM model achieved an accuracy of 54% on the test dataset. While this accuracy is relatively low, it serves as a baseline for further improvement through experimentation with advanced algorithms and feature engineering techniques.

## Project Structure

- `data/`: Directory containing the dataset files.
- `src/`: Source code directory.
  - `preprocessing.py`: Preprocessing script for data augmentation and PCA.
  - `svm_model.py`: Implementation of the SVM model.
- `notebooks/`: Jupyter notebooks for exploratory data analysis and model evaluation.

## Requirements

- Python 3.x
- scikit-learn
- OpenCV
- NumPy

## Usage

1. Clone this repository.
2. Install the required dependencies.
3. Execute the preprocessing script to augment the dataset and perform PCA.
4. Train the SVM model using the processed dataset.
5. Evaluate the model's performance using test data.

## References

- UCI Machine Learning Repository: [Burden Diseases Dataset](https://archive.ics.uci.edu/ml/datasets/burden+of+disease)
- scikit-learn Documentation: [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

## Contributors

- [Your Name](link_to_your_profile) - Project Lead and Developer

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
