"""examples for using pytorch"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def fizz_buzz(n:int) -> list[str]:
    """Given an integer n, return a string array (answer), where
    answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
    answer[i] == "Fizz" if i is divisible by 3.
    answer[i] == "Buzz" if i is divisible by 5.
    answer[i] == i if non of the above conditions are true.

    Parameters
    ----------
    n : int
        The maximum number of array

    Returns
    -------
    ans : list[str]
        A list containing strings following the above description
    """
    ans = []
    hash_map = {3 : "Fizz", 5 : "Buzz"}
    for num in range(1, n+1):
        string = ""
        for key, value in hash_map.items():
            if num % key == 0:
                string += value
        if not string:
            string = str(num)
        ans.append(string)
    return ans

def calculate_accuracy(y_pred: list[int], y_test: list[int]) -> float:
    """Calculate prediction accuracy

    Parameters
    ----------
    y_pred : list[int] or array_like
        A list containing predicted classes
    y_test : list[int] or array_like
        A list of ground truth classes

    Returns
    -------
    accuracy : float
        The accuracy of the predictions
    """
    correct_false_array = np.array(y_pred) == np.array(y_test)
    accuracy = np.sum(correct_false_array) / len(correct_false_array)
    return accuracy

def binary_to_list(binary:str, max_digit):
    binary_list = np.zeros(max_digit)
    for idx, value in enumerate(binary[-1::-1]):
        binary_list[idx] = float(value)
    return binary_list

def get_data(num_data):
    X = np.arange(1, num_data+1)
    X = np.array([binary_to_list(bin(x)[2:], 10) for x in X])
    X = torch.tensor(X, dtype=torch.float)
    y = fizz_buzz(num_data)
    string_list = ["Fizz", "Buzz", "FizzBuzz"]
    y = np.array([string if string in string_list else "same" for string in y])
    y = y[..., np.newaxis]
    enc = OneHotEncoder(handle_unknown='ignore').fit(y)
    y_one_hot = torch.tensor(enc.transform(y).toarray())
    classes = np.array(enc.categories_[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.1)
    #X_train = X[100:]
    #X_test = X[:100]
    #y_train = y_one_hot[100:]
    #y_test = y_one_hot[:100]
    return X_train, X_test, y_train, y_test, classes

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(10, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = F.softmax(self.fc2(x), dim=1) <- if we use crossentropy_loss, we don't need this
        x = self.fc2(x)
        return x

def binary_to_decimal(binary: str) -> int:
    """
    Convert a binary number to its decimal equivalent.

    Args:
        binary (str): A string representing the binary number.

    Returns:
        int: The decimal equivalent of the binary number.

    Raises:
        ValueError: If the input string contains characters other than '0' and '1'.
    """
    if not all(c in "01" for c in binary):
        raise ValueError("Invalid binary string")
    decimal = sum(int(digit) * 2 ** i for i, digit in enumerate(binary))
    return decimal

def train(X_train, y_train, model, epochs, optimizer, criterion):
    _, train_labels = torch.max(y_train, 1)
    for epoch in range(epochs):
        optimizer.zero_grad()   # zero the gradient buffers
        outputs = model(X_train)
        #print(outputs[:10])
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()    # Does the update
        if epoch % 10 == 0:
            _, predicted = torch.max(outputs, 1)
            print(f"Epoch = {epoch}, loss = {loss:2.5f}, accuracy = {calculate_accuracy(predicted.detach().numpy(), train_labels.detach().numpy()):1.5f}")

def main():

    num_data = 10000
    epochs = 1000
    X_train, X_test, y_train, y_test, classes = get_data(num_data)
    _, test_labels = torch.max(y_test, 1)

    # Define network
    model = Net()
    criterion = nn.CrossEntropyLoss()

    # create your optimizer
    optimizer = optim.Adam(model.parameters())
    train(X_train, y_train, model, epochs, optimizer, criterion)

    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        _, pred_tests = torch.max(outputs, 1)
        print(f"Test accuracy = {calculate_accuracy(pred_tests, test_labels)}")

    for idx, (x_bin, pred_test) in enumerate(zip(X_test, pred_tests)):
        x_bin_int_obj = list(map(int, x_bin.detach().numpy()))
        binary = "".join(list(map(str, x_bin_int_obj)))
        decimal = binary_to_decimal(binary)
        if classes[pred_test] == "same":
            print(f"({test_labels[idx]==pred_test}): Input:{decimal}, Output:{decimal}")
        else:
            print(f"({test_labels[idx]==pred_test}): Input:{decimal}, Output:{classes[pred_test]}")
if __name__ == "__main__":
    main()
