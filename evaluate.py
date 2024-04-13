import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.Model import mini_XCEPTION
from utils.dataset import FER2013

class ConfusionMatrixVisualizer:
    def __init__(self, label_names):
        """
        Initialize the ConfusionMatrix object with label names

        :param label_names: List of class labels
        """
        self.label_names = label_names
        self.num_classes = len(label_names)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predictions, true_labels):
        """
        Update the confusion matrix with predictions and true labels

        :param predictions: 1D vector of predicted labels, e.g., array([0,5,1,6,3,...], dtype=int64)
        :param true_labels: 1D vector of true labels, e.g., array([0,5,0,6,2,...], dtype=int64)
        """
        for prediction, true_label in zip(predictions, true_labels):
            self.matrix[prediction, true_label] += 1

    def draw(self):
        # Calculate the sum of each row for percentage calculation
        row_sums = self.matrix.sum(axis=1)
        # Convert counts to percentages
        for i in range(self.num_classes):
            self.matrix[i] = self.matrix[i] / row_sums[i]

        # Plot the normalized confusion matrix
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # Plot matrix without values
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.yticks(range(self.num_classes), self.label_names)  # Set Y-axis labels
        plt.xticks(range(self.num_classes), self.label_names, rotation=45)  # Set X-axis labels

        # Display numerical values within the matrix
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')

        # Adjust subplot parameters to fit the entire image area
        plt.tight_layout()
        # Add color bar
        plt.colorbar()
        # Save the confusion matrix as an image file, ensuring labels are fully displayed
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')
        plt.show()

def evaluate_model():
    confusion_matrix_visualizer = ConfusionMatrixVisualizer(labels_name=['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    total_test_loss = 0
    total_test_accuracy = 0
    sample_count = 0
    model.eval()  # Set model to evaluation mode

    # Iterate over the test dataset
    for index, (labels, images) in enumerate(test_loader):
        predictions = model(images.to(device))
        predicted_labels = np.argmax(predictions.cpu().detach().numpy(), axis=-1)
        true_labels = labels.numpy()
        confusion_matrix_visualizer.update(predicted_labels, true_labels)

        # Calculate accuracy
        accuracy = sum(predicted_labels == true_labels)
        # Calculate loss
        loss = loss_fn(predictions, labels.to(device))
        total_test_loss += loss.item()
        total_test_accuracy += accuracy
        sample_count += len(labels)

    # Calculate mean loss and accuracy
    average_test_loss = total_test_loss / sample_count
    average_test_accuracy = total_test_accuracy / sample_count
    print(f"Eval\tloss: {average_test_loss:.4f}\tacc: {average_test_accuracy:.4f}")
    confusion_matrix_visualizer.draw()

if __name__ == "__main__":
    num_workers = 0  # Number of subprocesses to use for data loading

    batch_size = 32
    input_size = (48, 48)
    num_classes = 7

    # Define model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = mini_XCEPTION(num_classes=num_classes)
    model.load_state_dict(torch.load("output/Epoch_200_emotion.pth", map_location=device))
    model.to(device)

    # Load test data
    test_dataset = FER2013("test", input_size=input_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Define loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Start evaluation
    evaluate_model()
