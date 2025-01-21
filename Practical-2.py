
import pandas as pd

# Step 1: Import data from a CSV file
from google.colab import files
uploaded = files.upload()

# Function to load CSV file using Pandas
def load_csv(filepath):
    return pd.read_csv(filepath)
# Load the dataset
myData = load_csv('sales_data.csv')
print(myData.head())

# Step 2: Perform basic data cleaning and add a new column for sales tax
def clean_and_modify_data(df, tax_rate=0.1):
    try:
        # Drop rows with missing values (if any)
        df.dropna(inplace=True)

        # Add a new column for sales tax (assuming a default tax rate of 10%)
        df['Sales Tax'] = df['Sales Amount'] * tax_rate

        # Add a new column for total amount (Sales + Tax)
        df['Total Amount'] = df['Sales Amount'] + df['Sales Tax']

        print("Data cleaned and modified successfully.")
        return df
    except KeyError as e:
        print(f"Error: Missing expected column. {e}")
        return None

# Step 3: Export the updated data back to a new CSV file
def export_data(df, updated_sales_data):
    try:
        df.to_csv(updated_sales_data, index=False)
        print(f"Updated data exported successfully to {updated_sales_data}.")
    except Exception as e:
        print(f"Error exporting data: {e}")

# Main execution block
if __name__ == "__main__":
    # Input and output file paths
    input_file = "sales_data.csv"  # Replace with the actual file path
    output_file = "updated_sales_data.csv"

    # Import data
    sales_data = load_csv(input_file)

    if sales_data is not None:
        # Perform data cleaning and modification
        updated_data = clean_and_modify_data(sales_data)

        if updated_data is not None:
            # Export updated data
            export_data(updated_data, output_file)
