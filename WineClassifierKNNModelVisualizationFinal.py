import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


##############################################################################################
# Wine Quality Classifier using K-Nearest Neighbors (KNN)
# -----------------------------------------------------------------------------------------
# This project classifies wine quality based on its chemical properties.
# We use the KNN algorithm and find the best value of K through hyperparameter tuning.
##############################################################################################


Border = "-" * 50


def WineClassifierKNN(DataPath):


    ##############################################################################################
    # Step 1 : Load the Dataset from CSV file
    ##############################################################################################

    print(Border)
    print("Step 1 : Load the Dataset from CSV file")
    print(Border)

    df = pd.read_csv(DataPath)

    print("Dataset loaded successfully.")
    print("Some entries from dataset :")
    print(df.head())
    print(Border)


    ##############################################################################################
    # Step 2 : Clean the Dataset by Removing Empty Rows
    ##############################################################################################

    print(Border)
    print("Step 2 : Clean the Dataset by Removing Empty Rows")
    print(Border)

    df.dropna(inplace=True)

    print("Total Records  :", df.shape[0])
    print("Total Columns  :", df.shape[1])
    print(Border)


    ##############################################################################################
    # Step 3 : Separate Independent and Dependent Variables
    # -----------------------------------------------------------------------------------------
    # X : Independent Variables (Features) — all columns except 'Class'
    # Y : Dependent Variable   (Label)    — 'Class' column (wine quality)
    ##############################################################################################

    print(Border)
    print("Step 3 : Separate Independent and Dependent Variables")
    print(Border)

    X = df.drop(columns="Class")   # Independent Variables (Features)
    Y = df["Class"]                # Dependent Variable   (Label)

    print("Shape of X (Features) :", X.shape)
    print("Shape of Y (Labels)   :", Y.shape)
    print("Input Columns         :", X.columns.tolist())
    print("Output Column         : Class")
    print(Border)


    ##############################################################################################
    # Step 4 : Split the Dataset into Training and Testing Sets
    # -----------------------------------------------------------------------------------------
    # Test Size     = 20%
    # Training Size = 80%
    # stratify=Y    ensures equal class distribution in both splits
    ##############################################################################################

    print(Border)
    print("Step 4 : Split the Dataset into Training and Testing Sets")
    print(Border)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

    print("Training and Testing split information :")
    print("X_train shape :", X_train.shape)
    print("X_test  shape :", X_test.shape)
    print("Y_train shape :", Y_train.shape)
    print("Y_test  shape :", Y_test.shape)
    print(Border)


    ##############################################################################################
    # Step 5 : Feature Scaling using Standard Scaler
    # -----------------------------------------------------------------------------------------
    # KNN is a distance-based algorithm.
    # Without scaling, features with larger values will dominate the distance calculation.
    # StandardScaler transforms data to have mean = 0 and standard deviation = 1.
    ##############################################################################################

    print(Border)
    print("Step 5 : Feature Scaling using Standard Scaler")
    print(Border)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)   # Fit on training data, then transform
    X_test_scaled  = scaler.transform(X_test)        # Only transform test data (no fit)

    print("Feature scaling completed successfully.")
    print(Border)


    ##############################################################################################
    # Step 6 : Hyperparameter Tuning — Finding the Best Value of K
    # -----------------------------------------------------------------------------------------
    # We test K values from 1 to 20.
    # For each K, we train the model and record the accuracy.
    # The K with the highest accuracy is selected as the best K.
    ##############################################################################################

    print(Border)
    print("Step 6 : Hyperparameter Tuning — Testing K values from 1 to 20")
    print(Border)

    accuracy_Scores = []
    k_values = range(1, 21)

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train_scaled, Y_train)
        Y_pred   = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracy_Scores.append(accuracy)

    print("Accuracy report for all K values (K = 1 to 20) :")
    for k, acc in zip(k_values, accuracy_Scores):
        print(f"  K = {k:2d}  -->  Accuracy : {acc * 100:.2f}%")
    print(Border)


    ##############################################################################################
    # Step 7 : Plot Graph — K Value vs Accuracy
    # -----------------------------------------------------------------------------------------
    # Visual representation helps identify the elbow point
    # where accuracy is highest and stabilizes.
    ##############################################################################################

    print(Border)
    print("Step 7 : Plot Graph — K Value vs Accuracy")
    print(Border)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_Scores, marker="o", color="steelblue", linewidth=2)
    plt.title("K Value vs Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(k_values))
    plt.tight_layout()
    plt.show()


    ##############################################################################################
    # Step 8 : Find the Best Value of K
    # -----------------------------------------------------------------------------------------
    # The K value that produced the highest accuracy is selected
    # as the optimal hyperparameter for the final model.
    ##############################################################################################

    print(Border)
    print("Step 8 : Find the Best Value of K")
    print(Border)

    best_k = list(k_values)[accuracy_Scores.index(max(accuracy_Scores))]
    print("Best value of K is :", best_k)
    print(Border)


    ##############################################################################################
    # Step 9 : Build the Final Model using the Best Value of K
    # -----------------------------------------------------------------------------------------
    # We re-train the KNN model using the optimal K found in Step 8.
    ##############################################################################################

    print(Border)
    print("Step 9 : Build the Final Model using Best K =", best_k)
    print(Border)

    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train_scaled, Y_train)
    Y_pred = final_model.predict(X_test_scaled)

    print("Final model trained successfully.")
    print(Border)


    ##############################################################################################
    # Step 10 : Calculate Final Accuracy
    ##############################################################################################

    print(Border)
    print("Step 10 : Calculate Final Accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test, Y_pred)
    print("Final Model Accuracy :", round(accuracy * 100, 2), "%")
    print(Border)


    ##############################################################################################
    # Step 11 : Display Confusion Matrix
    # -----------------------------------------------------------------------------------------
    # Confusion Matrix shows how many predictions were correct
    # and where the model made mistakes (per class).
    ##############################################################################################

    print(Border)
    print("Step 11 : Display Confusion Matrix")
    print(Border)

    cm = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix :\n", cm)
    print(Border)


    ##############################################################################################
    # Step 12 : Display Classification Report
    # -----------------------------------------------------------------------------------------
    # Provides Precision, Recall, and F1-Score for each class.
    ##############################################################################################

    print(Border)
    print("Step 12 : Display Classification Report")
    print(Border)

    print(classification_report(Y_test, Y_pred))
    print(Border)


##############################################################################################
# Main Entry Point
##############################################################################################

def main():
    print(Border)
    print("Wine Quality Classifier using K-Nearest Neighbors (KNN)")
    print(Border)

    WineClassifierKNN("WinePredictor.csv")


if __name__ == "__main__":
    main()