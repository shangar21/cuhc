# Example: customer segments

`cluster_customers.py` clusters the 200 customers of the Kaggle "Mall Customers" data
set (`vjchoudhary7/customer-segmentation-tutorial-in-python`) by annual income and
spending score, with cuhc and with scikit-learn. It draws one row of plots for each
library: the Ward dendrogram, the five Ward segments and the HDBSCAN clusters.

The tutorial in the documentation (`docs/tutorial_customer_segments.md`) explains each
step.

## Steps

1. Download `Mall_Customers.csv` from the Kaggle data set into `~/Downloads`.
2. Install cuhc, and the packages that draw the figure:

   ```bash
   pip install matplotlib scipy
   ```

3. Run the script:

   ```bash
   python cluster_customers.py
   ```

   To write the figure to a file, use `--save customers.png`. If the CSV file is in a
   different directory, use `--data path/to/Mall_Customers.csv`.
