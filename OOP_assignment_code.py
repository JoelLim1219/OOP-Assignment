import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class TransactionPipeline:
    
	# Initialize file paths and data frames
    def __init__(self, transactions_path, merchant_codes_path, fraud_path):
        self.transactions_path = transactions_path
        self.merchant_codes_path = merchant_codes_path
        self.fraud_path = fraud_path

        self.df_raw = None
        self.df_clean = None
        self.merged_df = None

	# Method to load CSV file and convert columnc to numeric
    def load_data(self):
        
        # Read the CSV files into pandas DataFrames
        self.df_raw = pd.read_csv(self.transactions_path)
        merchant_df = pd.read_csv(self.merchant_codes_path)
        fraud_df = pd.read_csv(self.fraud_path)

		# Convert mcc and id to numeric, 'erros = "coerce"' deals with the erros by coercing errors to NaN
        merchant_df["mcc"] = pd.to_numeric(merchant_df["mcc"], errors="coerce")
        fraud_df["id"] = pd.to_numeric(fraud_df["id"], errors="coerce")

        return merchant_df, fraud_df

	# Method to filter transactions
    def clean_transactions(self):
        df = self.df_raw.copy()

		# Section 1: Data cleaning on Transaction records
        # S1 a) Remove transactions with any error value
        # Convert error column to pandas string type and strip whitespace
        errors_clean = df["errors"].astype("string").str.strip()
        # Keep rows where error column is empty, isna() checks for missing values while erros_clean checks for empty string
        # Row is kept if either condition is true
        df = df[errors_clean.isna() | (errors_clean == "")]

        # Clean amount from string to numeric
        df["amount"] = pd.to_numeric(
            df["amount"]
            .astype("string") # Convert to string
            .str.replace("$", "", regex=False) # Remove $ symbol
            .str.replace(",", "", regex=False) # Remove commas
            .str.strip(), # Remove leading/trailing whitespace
            errors="coerce",
        )

        # S1 b) Remove missing id, mcc, amount
        # Filter out rows with missing data on id, mcc, or amount columns
        df = df.dropna(subset=["id", "mcc", "amount"])

        # S1 c) Remove amount == 0
        # Rows with non-zero amount will be retained
        df = df[df["amount"] != 0]

        # S1 d) Drop duplicates on (id, mcc, amount), keep latest by date
        # Convert date column to datetime format
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Dataframe sorted by date, duplicates are removed based on id, mcc and amount to only remain the most recent transaction
        df = df.sort_values("date").drop_duplicates(
            subset=["id", "mcc", "amount"],
            keep="last"
        ).reset_index(drop=True) # Reset the index to form clean sequence after dropping duplicates

		# Change ID and mcc to integer type after cleaning since the missing values are removed
        df["id"] = df["id"].astype("int64")
        df["mcc"] = df["mcc"].astype("int64")

        self.df_clean = df # Store the filtered dataframe in the class attribute

    def cleaning_summary(self):
        return {
            "Total transactions before data cleaning": len(self.df_raw),
            "Total transactions after data cleaning": len(self.df_clean),
        }

	# Method to merge and manage data
    def merge_data(self, merchant_df, fraud_df):
        
		# Section 2: Data merging and management
        # S2 a) Merge transactions with merchant codes and fraud status
        merged_df = (
            self.df_clean
            .merge(merchant_df, on="mcc", how="left") # Merge with 'mcc' as the key, left join to keep all transactions
            .merge(fraud_df, on="id", how="left") # Merge with 'id' as the key, left join to keep all transactions
        )

		# S2 b) Handle missing values in Category, Business_Type, Fraud_Status
        for col in ["Category", "Business_Type", "Fraud_Status"]:
            merged_df[col] = merged_df[col].fillna("Unknown").astype("string").str.strip() # Fill missing values with "Unknown"
            merged_df[col] = merged_df[col].replace("", "Unknown") # Replace empty strings with "Unknown"

        # merged_df["Fraud_Status"] = merged_df["Fraud_Status"].str.title() # Convert values under Fraud_Status column to title case (e.g. "yes" to "Yes")
        # merged_df["Fraud_Status"] = merged_df["Fraud_Status"].where(
        #     merged_df["Fraud_Status"].isin(["Yes", "No"]),
        #     "Unknown"
        # )

        # merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
        
        self.merged_df = merged_df

    def merging_summary(self):
        df = self.merged_df

		# S2 c) Interpretation of amount column 
        # Obtain the count of debit and credit transactions
        debit_count = (df["amount"] < 0).sum()
        credit_count = (df["amount"] >= 0).sum()

		# Obtain the count of transactions by fraud status
        fraudulent_count = (df["Fraud_Status"] == "Yes").sum()
        legitimate_count = (df["Fraud_Status"] == "No").sum()
        unknown_count = (df["Fraud_Status"] == "Unknown").sum()

		# Obtain the count of unique business types in total and by fraud status
        business_types_total = df["Business_Type"].nunique(dropna=True)
        business_types_fraud = df.loc[df["Fraud_Status"] == "Yes", "Business_Type"].nunique(dropna=True)
        business_types_legit = df.loc[df["Fraud_Status"] == "No", "Business_Type"].nunique(dropna=True)
        business_types_unknown = df.loc[df["Fraud_Status"] == "Unknown", "Business_Type"].nunique(dropna=True)

		# S2 d) Summary of merged data
        return {
            "Total transactions": len(df),
            "Number of debit transactions": int(debit_count),
            "Number of credit transactions": int(credit_count),
            "Number of fraudulent transactions": int(fraudulent_count),
            "Number of legitimate transactions": int(legitimate_count),
            "Number of unknown transactions": int(unknown_count),
            "Number of business types in transactions": int(business_types_total),
            "Number of business types involved in fraudulent transactions": int(business_types_fraud),
            "Number of business types involved in legitimate transactions": int(business_types_legit),
            "Number of business types involved in unknown transactions": int(business_types_unknown),
        }

	# Method to run the entire pipeline
    def run(self):
        merchant_df, fraud_df = self.load_data() # Load data
        self.clean_transactions() # Clean transactions
        self.merge_data(merchant_df, fraud_df) # Merge, manage and print data summaries


class DataAnalytics:
    def __init__(self, merged_df):
        self.df = merged_df.copy()

	# Section 3: Data analytics and visualization
    # S3 a) Average transaction amount per date
    def avg_transaction_amount_per_date(self):
        
        out = (
            self.df.groupby(self.df["date"].dt.date)["amount"] # Group by date
            .mean() # Calculate mean for each date group
            .reset_index(name="avg_amount") # Reset index to turn the date back into a column and name the average amount column as "avg_amount"
        )
        
        return out

	# S3 b) Fraud rate (%) by hour of day
    def fraud_rate_by_hour(self):
        
        out = (
            self.df.assign(hour=self.df["date"].dt.hour) # Assign an additional column "hour" by extracting hour from date
            .groupby("hour") # Group by hour
            .agg( # Compute total transaction and fraud transaction counts for each hour group
                total_transactions=("id", "size"), # Count total transactions by ID for each hour group
                fraud_transactions=("Fraud_Status", lambda s: (s == "Yes").sum()), # Count number of transaction where "Fraud_Status" is "Yes" for each hour group
            )
            .reset_index() # Turns grouped index back to regular column
        )
        
        out["fraud_rate"] = (out["fraud_transactions"] / out["total_transactions"]) * 100
        
        return out

	# S3 c) Overall fraud status distribution
    def fraud_status_distribution(self):
        
        out = (
            self.df["Fraud_Status"]
            # .fillna("Unknown")
            .value_counts() # Count the occurrences of each unique value in "Fraud_Status" column
            .reindex(["Yes", "No", "Unknown"], fill_value=0) # Reorder the index to have "Yes", "No", "Unknown" in that order, fill missing values with 0
            .reset_index()
        )
        
        out.columns = ["Fraud_Status", "count"] # Rename dataframe columns to "Fraud_Status" and "count"
        
        out["percentage"] = (out["count"] / out["count"].sum()) * 100 # Calculate percentage for each fraud status category
        
		# Map the fraud status to more descriptive labels for visualization
        out["label"] = out["Fraud_Status"].map({
            "Yes": "Fraud",
            "No": "Legitimate",
            "Unknown": "Unknown",
        })
        
        return out

	# S3 d) Debit vs credit transactions by fraud status
    def debit_credit_by_fraud_status(self):
        
        df = self.df.copy()
        
		# Create a new column "payment_type" to classify transactions based on the amount value
        df["payment_type"] = df["amount"].apply(lambda x: "Debit" if x < 0 else "Credit")

		# Group by "Fraud_Status" and "payment_type" and get the count for each combination
        summary = (
            df.groupby(["Fraud_Status", "payment_type"])
            .size()
            .reset_index(name="count")
        )

        status_order = ["Yes", "No", "Unknown"]
        payment_order = ["Debit", "Credit"]

		# Ensure the rows and column appear in specified order
        summary["Fraud_Status"] = pd.Categorical(summary["Fraud_Status"], categories=status_order, ordered=True)
        summary["payment_type"] = pd.Categorical(summary["payment_type"], categories=payment_order, ordered=True)

		# Create pivot table where "Fraud_Status" is the index, "payment_type" is the columns and "count" is the values and fill the missing values with 0
        pivot_counts = summary.pivot(index="Fraud_Status", columns="payment_type", values="count").fillna(0)
        
        # pivot_counts = pivot_counts.reindex(status_order).fillna(0)

        # for col in payment_order:
        #     if col not in pivot_counts.columns:
        #         pivot_counts[col] = 0

        # pivot_counts = pivot_counts[payment_order] # Reorder the columns
        
		# Calculate the percentage for each payment type within each fraud status category
        pivot_pct = pivot_counts.div(pivot_counts.sum(axis=1), axis=0) * 100

        return pivot_counts, pivot_pct

	# S3 e) Top 5 business types by total fraudulent transaction amount
    def top5_business_types_by_fraud_amount(self):
        
		# Filter by fraudulent transactions, group by business type, get the sum of each business type, sort and get the top 5 business types
        out = (
            self.df[self.df["Fraud_Status"] == "Yes"]
            .groupby("Business_Type")["amount"]
            .sum()
            .abs()
            .sort_values(ascending=False)
            .head(5)
            .reset_index(name="total_fraud_amount")
        )
        
        return out


class UIDashboard(tk.Tk):
    def __init__(self, pipeline):
        super().__init__()
        self.title("Data Analytics Dashboard")
        self.geometry("1200x760")

        self.pipeline = pipeline
        self.analytics = DataAnalytics(self.pipeline.merged_df)

        self._build_layout()
        self._show_summaries()
        self.plot_selected("3a")

    def _build_layout(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.summary_text = tk.Text(top_frame, height=12, wrap=tk.WORD)
        self.summary_text.pack(fill=tk.X)

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="Choose chart:").pack(side=tk.LEFT, padx=(0, 8))

        self.chart_var = tk.StringVar(value="3a")
        chart_options = [
            ("3a - Average Transaction Amount per Date", "3a"),
            ("3b - Fraud Rate by Hour", "3b"),
            ("3c - Fraud Status Distribution", "3c"),
            ("3d - Debit vs Credit by Fraud Status", "3d"),
            ("3e - Top 5 Business Types by Fraud Amount", "3e"),
        ]

        for label, value in chart_options:
            ttk.Radiobutton(
                control_frame,
                text=label,
                value=value,
                variable=self.chart_var,
                command=lambda v=value: self.plot_selected(v)
            ).pack(side=tk.LEFT, padx=5)

        chart_frame = ttk.Frame(self)
        chart_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = plt.Figure(figsize=(12, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _show_summaries(self):
        csum = self.pipeline.cleaning_summary()
        msum = self.pipeline.merging_summary()

        lines = []
        lines.append("=== DATA CLEANING SUMMARY ===")
        for k, v in csum.items():
            lines.append(f"{k}: {v}")

        lines.append("")
        lines.append("=== DATA MERGING SUMMARY ===")
        lines.append(f"Total transactions: {msum['Total transactions']}")
        lines.append(f"Number of debit transactions: {msum['Number of debit transactions']}")
        lines.append(f"Number of credit transactions: {msum['Number of credit transactions']}")
        lines.append("Fraud status breakdown:")
        lines.append(f"  Fraudulent: {msum['Number of fraudulent transactions']}")
        lines.append(f"  Legitimate: {msum['Number of legitimate transactions']}")
        lines.append(f"  Unknown: {msum['Number of unknown transactions']}")
        lines.append("Business type summary:")
        lines.append(f"  In transactions: {msum['Number of business types in transactions']}")
        lines.append(f"  In fraudulent transactions: {msum['Number of business types involved in fraudulent transactions']}")
        lines.append(f"  In legitimate transactions: {msum['Number of business types involved in legitimate transactions']}")
        lines.append(f"  In unknown transactions: {msum['Number of business types involved in unknown transactions']}")

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, "\n".join(lines))
        self.summary_text.configure(state=tk.DISABLED)

    def _annotate_bars(self, bars, labels):
        for bar, label in zip(bars, labels):
            y = bar.get_height()
            self.ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, y),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=8,
            )

    def plot_selected(self, chart_key):
        self.ax.clear()

        if chart_key == "3a":
            data = self.analytics.avg_transaction_amount_per_date()
            bars = self.ax.bar(data["date"].astype(str), data["avg_amount"], color="#2E86AB")
            self.ax.set_title("Average Transaction Amount ($) per Date")
            self.ax.set_xlabel("Date")
            self.ax.set_ylabel("Average Amount ($)")
            self.ax.tick_params(axis="x", rotation=45)
            self._annotate_bars(bars, [f"${v:.2f}" for v in data["avg_amount"]])

        elif chart_key == "3b":
            data = self.analytics.fraud_rate_by_hour()
            bars = self.ax.bar(data["hour"].astype(str), data["fraud_rate"], color="#E76F51")
            self.ax.set_title("Fraud Rate (%) by Hour of Day")
            self.ax.set_xlabel("Hour of Day")
            self.ax.set_ylabel("Fraud Rate (%)")
            ymax = max(data["fraud_rate"].max() * 1.2, 1)
            self.ax.set_ylim(0, ymax)
            labels = [
                f"{r:.2f}%\n(n={f}/{t})"
                for r, f, t in zip(data["fraud_rate"], data["fraud_transactions"], data["total_transactions"])
            ]
            self._annotate_bars(bars, labels)

        elif chart_key == "3c":
            data = self.analytics.fraud_status_distribution()
            bars = self.ax.bar(data["label"], data["count"], color=["#D62828", "#2A9D8F", "#8D99AE"])
            self.ax.set_title("Overall Fraud Status Distribution")
            self.ax.set_xlabel("Fraud Status")
            self.ax.set_ylabel("Transactions Count")
            labels = [f"{c}\n({p:.1f}%)" for c, p in zip(data["count"], data["percentage"])]
            self._annotate_bars(bars, labels)

        elif chart_key == "3d":
            counts, pct = self.analytics.debit_credit_by_fraud_status()
            x = list(range(len(counts.index)))
            width = 0.35

            debit_bars = self.ax.bar(
                [i - width / 2 for i in x], pct["Debit"], width=width, label="Debit", color="#264653"
            )
            credit_bars = self.ax.bar(
                [i + width / 2 for i in x], pct["Credit"], width=width, label="Credit", color="#F4A261"
            )

            self.ax.set_title("Debit vs Credit Transactions by Fraud Status")
            self.ax.set_xlabel("Fraud Status")
            self.ax.set_ylabel("Percentage of Transactions (%)")
            self.ax.set_xticks(x)
            self.ax.set_xticklabels(["Fraud", "Legitimate", "Unknown"])
            self.ax.legend()

            for i, status in enumerate(counts.index):
                debit_label = f"{int(counts.loc[status, 'Debit'])}\n({pct.loc[status, 'Debit']:.1f}%)"
                credit_label = f"{int(counts.loc[status, 'Credit'])}\n({pct.loc[status, 'Credit']:.1f}%)"
                self.ax.annotate(debit_label, (i - width / 2, pct.loc[status, "Debit"]),
                                 ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=8)
                self.ax.annotate(credit_label, (i + width / 2, pct.loc[status, "Credit"]),
                                 ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=8)

        elif chart_key == "3e":
            data = self.analytics.top5_business_types_by_fraud_amount()
            if data.empty:
                self.ax.text(0.5, 0.5, "No fraudulent transactions found.", ha="center", va="center")
                self.ax.set_title("Top 5 Business Types by Fraudulent Transaction Amount ($)")
                self.ax.set_axis_off()
            else:
                bars = self.ax.bar(data["Business_Type"], data["total_fraud_amount"], color="#6A994E")
                self.ax.set_title("Top 5 Business Types by Fraudulent Transaction Amount ($)")
                self.ax.set_xlabel("Business Type")
                self.ax.set_ylabel("Total Fraudulent Amount ($)")
                self.ax.tick_params(axis="x", rotation=25)
                self._annotate_bars(bars, [f"${v:.2f}" for v in data["total_fraud_amount"]])

        self.fig.tight_layout()
        self.canvas.draw()


def main():
    
    pipeline = TransactionPipeline(
        transactions_path="transactions.csv",
        merchant_codes_path="merchant_codes.csv",
        fraud_path="fraud.csv"
    )
    pipeline.run()

    app = UIDashboard(pipeline)
    app.mainloop()


main()