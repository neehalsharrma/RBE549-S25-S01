import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = "loss_log.csv"
df = pd.read_csv(file_path)

print(df.columns)
print(df.head())
df.columns = df.columns.str.strip()


# Extract columns
epochs = df["epoch"]
test_loss = df["test_loss"]
train_loss = df["train_loss"]

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, test_loss, label="Test Loss", color='red', marker='o')
plt.plot(epochs, train_loss, label="Train Loss", color='blue', marker='s')

# Labels and title
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Testing Loss vs. Epoch")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
