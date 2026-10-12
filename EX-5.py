{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN3cvn5+JcrwsaAqfGZ7bF3",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kabianjackie10-bot/MLA0201--fundamentals-of-machine-learning/blob/main/EX-5.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Cpn1Tl6Peqzy"
      },
      "outputs": [],
      "source": [
        "\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay\n",
        "\n",
        "# 1. Load dataset\n",
        "iris = load_iris()\n",
        "X = iris.data[:, :2]  # Use two features for visualization\n",
        "y = iris.target\n",
        "\n",
        "# 2. Split the dataset\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=0.25, random_state=42, stratify=y\n",
        ")\n",
        "\n",
        "# 3. Standardize features\n",
        "scaler = StandardScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)\n",
        "\n",
        "# 4. Train the K-NN model\n",
        "model = KNeighborsClassifier(n_neighbors=5)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# 5. Test the model\n",
        "y_pred = model.predict(X_test)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "\n",
        "print(\"K-Nearest Neighbours Classification\")\n",
        "print(\"-----------------------------------\")\n",
        "print(\"Number of neighbours (K): 5\")\n",
        "print(f\"Test Accuracy: {accuracy * 100:.2f}%\")\n",
        "\n",
        "print(\"\\nSample predictions:\")\n",
        "for actual, predicted in zip(y_test[:5], y_pred[:5]):\n",
        "    print(f\"Actual: {iris.target_names[actual]}, \"\n",
        "          f\"Predicted: {iris.target_names[predicted]}\")\n",
        "\n",
        "# 6. Plot classification regions\n",
        "x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5\n",
        "y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5\n",
        "\n",
        "xx, yy = __import__(\"numpy\").meshgrid(\n",
        "    __import__(\"numpy\").arange(x_min, x_max, 0.03),\n",
        "    __import__(\"numpy\").arange(y_min, y_max, 0.03)\n",
        ")\n",
        "\n",
        "grid = __import__(\"numpy\").c_[xx.ravel(), yy.ravel()]\n",
        "Z = model.predict(grid).reshape(xx.shape)\n",
        "\n",
        "plt.figure(figsize=(9, 5))\n",
        "plt.contourf(xx, yy, Z, alpha=0.25, cmap=\"viridis\")\n",
        "plt.scatter(\n",
        "    X_train[:, 0], X_train[:, 1],\n",
        "    c=y_train, cmap=\"viridis\", edgecolors=\"black\",\n",
        "    label=\"Training samples\"\n",
        ")\n",
        "plt.xlabel(\"Standardized Sepal Length\")\n",
        "plt.ylabel(\"Standardized Sepal Width\")\n",
        "plt.title(\"K-NN Classification Regions (K = 5)\")\n",
        "plt.show()\n",
        "\n",
        "# 7. Plot confusion matrix\n",
        "ConfusionMatrixDisplay.from_predictions(\n",
        "    y_test, y_pred, display_labels=iris.target_names,\n",
        "    cmap=\"Blues\"\n",
        ")\n",
        "plt.title(\"K-NN Confusion Matrix\")\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
      ]
    }
  ]
}