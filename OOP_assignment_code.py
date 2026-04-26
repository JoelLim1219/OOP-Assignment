import tkinter as tk
from tkinter import ttk
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


plt.rcParams["font.family"] = "Times New Roman"


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
            .apply(lambda s: s.abs().sum())
            .sort_values(ascending=False)
            .head(5)
            .reset_index(name="total_fraud_amount")
        )
        
        return out


class AnalyticsDashboardUI:
    def __init__(self, analytics):
        self.analytics = analytics
        self.title_size = 16
        self.axis_title_size = 13
        self.tick_size = 11
        self.label_size = 10

        self.fig = None
        self.ax_3a = None
        self.ax_3b = None
        self.ax_3c = None
        self.ax_3d = None
        self.ax_3e = None

        self.root = None
        self.canvas = None
        self.canvas_window = None
        self.fig_canvas = None

    def run(self):
        self._create_figure()
        self._plot_all_sections()
        self._build_window()
        self.root.mainloop()

    @staticmethod
    def _annotate_bars(ax, bars, labels, fontsize=9, y_offset=4):
        for bar, label in zip(bars, labels):
            y = bar.get_height()
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, y),
                ha="center",
                va="bottom",
                xytext=(0, y_offset),
                textcoords="offset points",
                fontsize=fontsize,
            )

    def _create_figure(self):
        self.fig, axes = plt.subplots(5, 1, figsize=(16, 30))
        self.ax_3a, self.ax_3b, self.ax_3c, self.ax_3d, self.ax_3e = axes

    def _plot_all_sections(self):
        self._plot_3a()
        self._plot_3b()
        self._plot_3c()
        self._plot_3d()
        self._plot_3e()

        self.fig.suptitle("Fraud Analytics Overview", fontsize=18)
        self.fig.tight_layout(rect=[0, 0, 1, 0.98])

    def _plot_3a(self):
        data_3a = self.analytics.avg_transaction_amount_per_date()
        x_positions = list(range(len(data_3a)))
        self.ax_3a.plot(
            x_positions,
            data_3a["avg_amount"],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=5,
            color="#1D4ED8",
        )
        self.ax_3a.set_title("3a) Average Transaction Amount ($) per Date", fontsize=self.title_size)
        self.ax_3a.set_xlabel("Date", fontsize=self.axis_title_size)
        self.ax_3a.set_ylabel("Average Amount ($)", fontsize=self.axis_title_size)
        tick_step = max(1, len(x_positions) // 12)
        tick_positions = x_positions[::tick_step]
        tick_labels = data_3a["date"].astype(str).iloc[::tick_step]
        self.ax_3a.set_xticks(tick_positions)
        self.ax_3a.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=self.tick_size)
        self.ax_3a.tick_params(axis="y", labelsize=self.tick_size)
        if x_positions:
            self.ax_3a.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
        self.ax_3a.margins(x=0.02)
        self.ax_3a.set_ylim(30, 50)
        self.ax_3a.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        for x, y in zip(x_positions, data_3a["avg_amount"]):
            self.ax_3a.annotate(
                f"${y:.2f}",
                (x, y),
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                fontsize=self.label_size,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
            )

    def _plot_3b(self):
        data_3b = self.analytics.fraud_rate_by_hour()
        x_positions = list(range(len(data_3b)))
        bars_3b = self.ax_3b.bar(x_positions, data_3b["fraud_rate"], color="#E76F51")
        self.ax_3b.set_title("3b) Fraud Rate (%) by Hour of Day", fontsize=self.title_size)
        self.ax_3b.set_xlabel("Hour of Day", fontsize=self.axis_title_size)
        self.ax_3b.set_ylabel("Fraud Rate (%)", fontsize=self.axis_title_size)
        self.ax_3b.set_xticks(x_positions)
        self.ax_3b.set_xticklabels(data_3b["hour"].astype(str), rotation=0, fontsize=self.tick_size)
        self.ax_3b.tick_params(axis="y", labelsize=self.tick_size)
        if x_positions:
            self.ax_3b.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
        self.ax_3b.margins(x=0.02)
        ymax = max(data_3b["fraud_rate"].max() * 1.2, 1)
        self.ax_3b.set_ylim(0, ymax)
        self.ax_3b.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        self._annotate_bars(
            self.ax_3b,
            bars_3b,
            [f"{r:.2f}%" for r in data_3b["fraud_rate"]],
            fontsize=self.label_size,
        )

    def _plot_3c(self):
        data_3c = self.analytics.fraud_status_distribution()
        pie_colors = ["#D62828", "#2A9D8F", "#8D99AE"]
        display_counts = data_3c["count"].astype(float).copy()
        total_count = float(display_counts.sum())
        fraud_mask = data_3c["label"] == "Fraud"
        if total_count > 0 and fraud_mask.any():
            fraud_idx = data_3c.index[fraud_mask][0]
            fraud_value = display_counts.loc[fraud_idx]
            min_visible_fraction = 0.02
            min_visible_value = total_count * min_visible_fraction
            if 0 < fraud_value < min_visible_value:
                delta = min_visible_value - fraud_value
                donor_idx = display_counts.drop(index=fraud_idx).idxmax()
                display_counts.loc[fraud_idx] = min_visible_value
                display_counts.loc[donor_idx] = max(1.0, display_counts.loc[donor_idx] - delta)

        explode = [0.16 if label == "Fraud" else 0.0 for label in data_3c["label"]]
        wedges, _ = self.ax_3c.pie(
            display_counts,
            labels=None,
            colors=pie_colors,
            startangle=90,
            explode=explode,
            wedgeprops={"edgecolor": "white", "linewidth": 1.4},
        )

        for wedge, label, count, pct in zip(
            wedges, data_3c["label"], data_3c["count"], data_3c["percentage"]
        ):
            theta = (wedge.theta1 + wedge.theta2) / 2.0
            x = math.cos(math.radians(theta))
            y = math.sin(math.radians(theta))
            ha = "left" if x >= 0 else "right"
            label_text = f"{label}: {count} ({pct:.1f}%)"
            label_radius = 1.25
            label_y_adjust = 0.0
            label_x = label_radius * x
            label_y = (label_radius * y) + label_y_adjust
            if label == "Fraud":
                label_x = 0.52
                label_y = 1.06
                ha = "left"
            self.ax_3c.annotate(
                label_text,
                xy=(1.02 * x, 1.02 * y),
                xytext=(label_x, label_y),
                ha=ha,
                va="center",
                fontsize=self.label_size,
                arrowprops={"arrowstyle": "-", "color": "#666666", "linewidth": 1.0},
            )
        self.ax_3c.set_title("3c) Overall Fraud Status Distribution", fontsize=self.title_size)
        self.ax_3c.axis("equal")

    def _plot_3d(self):
        counts, pct = self.analytics.debit_credit_by_fraud_status()
        x = list(range(len(counts.index)))
        width = 0.35
        bars_debit = self.ax_3d.bar(
            [i - width / 2 for i in x], pct["Debit"], width=width, label="Debit", color="#2563EB"
        )
        bars_credit = self.ax_3d.bar(
            [i + width / 2 for i in x], pct["Credit"], width=width, label="Credit", color="#0EA5A4"
        )
        self.ax_3d.set_title("3d) Debit vs Credit Transactions by Fraud Status", fontsize=self.title_size)
        self.ax_3d.set_xlabel("Fraud Status", fontsize=self.axis_title_size)
        self.ax_3d.set_ylabel("Percentage of Transactions (%)", fontsize=self.axis_title_size)
        self.ax_3d.set_xticks(x)
        self.ax_3d.set_xticklabels(["Fraud", "Legitimate", "Unknown"], rotation=0, fontsize=self.tick_size)
        self.ax_3d.tick_params(axis="y", labelsize=self.tick_size)
        max_pct_value = float(pct.to_numpy().max()) if not pct.empty else 100.0
        self.ax_3d.set_ylim(0, min(120, max(105, max_pct_value + 12)))
        self.ax_3d.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        self.ax_3d.legend(fontsize=self.tick_size)
        for i, status in enumerate(counts.index):
            debit_label = f"{int(counts.loc[status, 'Debit'])}\n({pct.loc[status, 'Debit']:.1f}%)"
            credit_label = f"{int(counts.loc[status, 'Credit'])}\n({pct.loc[status, 'Credit']:.1f}%)"
            self.ax_3d.annotate(
                debit_label,
                (i - width / 2, pct.loc[status, "Debit"]),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=self.label_size,
            )
            self.ax_3d.annotate(
                credit_label,
                (i + width / 2, pct.loc[status, "Credit"]),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=self.label_size,
            )

    def _plot_3e(self):
        data_3e = self.analytics.top5_business_types_by_fraud_amount()
        if data_3e.empty:
            self.ax_3e.text(0.5, 0.5, "No fraudulent transactions found.", ha="center", va="center")
            self.ax_3e.set_title("3e) Top 5 Business Types by Fraudulent Transaction Amount ($)", fontsize=self.title_size)
            self.ax_3e.set_axis_off()
            return

        bars_3e = self.ax_3e.bar(data_3e["Business_Type"], data_3e["total_fraud_amount"], color="#6A994E")
        self.ax_3e.set_title("3e) Top 5 Business Types by Fraudulent Transaction Amount ($)", fontsize=self.title_size)
        self.ax_3e.set_xlabel("Business Type", fontsize=self.axis_title_size)
        self.ax_3e.set_ylabel("Total Fraudulent Amount ($)", fontsize=self.axis_title_size)
        self.ax_3e.tick_params(axis="x", rotation=0, labelsize=self.tick_size)
        self.ax_3e.tick_params(axis="y", labelsize=self.tick_size)
        max_value = float(data_3e["total_fraud_amount"].max())
        upper_limit = max(5000, math.ceil((max_value * 1.12) / 500) * 500)
        self.ax_3e.set_ylim(0, upper_limit)
        self.ax_3e.set_yticks(range(0, int(upper_limit) + 1, 500))
        self.ax_3e.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.35)
        self._annotate_bars(
            self.ax_3e,
            bars_3e,
            [f"${v:.2f}" for v in data_3e["total_fraud_amount"]],
            fontsize=self.label_size,
        )

    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Fraud Analytics Dashboard")
        self.root.geometry("1400x900")

        style = ttk.Style(self.root)
        style.configure(".", font=("Times New Roman", 11))

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, highlightthickness=0)
        v_scroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=v_scroll.set)

        v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=inner, anchor="nw")

        inner.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._fit_figure_to_width)

        self.fig_canvas = FigureCanvasTkAgg(self.fig, master=inner)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(fill="x", anchor="nw")
        self._update_scrollregion()

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_scrollregion(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_figure_to_width(self, event):
        self.canvas.itemconfigure(self.canvas_window, width=event.width)
        width_inches = max(10.0, (event.width - 30) / self.fig.dpi)
        aspect_ratio = 30 / 16
        self.fig.set_size_inches(width_inches, width_inches * aspect_ratio, forward=True)
        self.fig.tight_layout(rect=[0, 0, 1, 0.98])
        self.fig_canvas.draw_idle()
        self._update_scrollregion()

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_close(self):
        self.canvas.unbind_all("<MouseWheel>")
        plt.close(self.fig)
        self.root.quit()
        self.root.destroy()


class ReportPrinter:
    def __init__(self, pipeline, analytics):
        self.pipeline = pipeline
        self.analytics = analytics

    def print_all_reports(self):
        self.print_section_summaries()
        self.print_section3_tables()

    def print_section_summaries(self):
        cleaning = self.pipeline.cleaning_summary()
        merging = self.pipeline.merging_summary()

        print("\n=== SECTION 1: DATA CLEANING SUMMARY ===")
        print(f"Total transactions before data cleaning = {cleaning['Total transactions before data cleaning']}")
        print(f"Total transactions after data cleaning = {cleaning['Total transactions after data cleaning']}")

        print("\n=== SECTION 2: DATA MERGING SUMMARY ===")
        print("Transactions summary after data merging:")
        print("-" * 50)
        print(f"Total transactions = {merging['Total transactions']}")
        print(f"Number of debit transactions = {merging['Number of debit transactions']}")
        print(f"Number of credit transactions = {merging['Number of credit transactions']}")
        print("Fraud status breakdown:")
        print(f"Number of fraudulent transactions = {merging['Number of fraudulent transactions']}")
        print(f"Number of legitimate transactions = {merging['Number of legitimate transactions']}")
        print(f"Number of unknown transactions = {merging['Number of unknown transactions']}")
        print("Business type summary:")
        print(f"Number of business types in transactions = {merging['Number of business types in transactions']}")
        print(f"Number of business types involved in fraudulent transactions: {merging['Number of business types involved in fraudulent transactions']}")
        print(f"Number of business types involved in legitimate transactions: {merging['Number of business types involved in legitimate transactions']}")
        print(f"Number of business types involved in unknown transactions: {merging['Number of business types involved in unknown transactions']}")

    @staticmethod
    def _format_table_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    def print_ascii_table(self, title, dataframe):
        print(f"\n{title}")
        if dataframe.empty:
            print("+-------------------------+")
            print("| No rows to display.     |")
            print("+-------------------------+")
            return

        headers = [str(col) for col in dataframe.columns]
        rows = [[self._format_table_value(val) for val in row] for row in dataframe.to_numpy()]
        widths = []
        for idx, header in enumerate(headers):
            max_row_width = max(len(row[idx]) for row in rows) if rows else 0
            widths.append(max(len(header), max_row_width))

        separator = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"

        print(separator)
        print(header_line)
        print(separator)
        for row in rows:
            row_line = "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"
            print(row_line)
        print(separator)

    def print_section3_tables(self):
        print("\n=== SECTION 3: DATA ANALYSIS TABLES ===")

        table_3a = self.analytics.avg_transaction_amount_per_date().copy()
        table_3a["date"] = table_3a["date"].astype(str)
        self.print_ascii_table("3a) Average Transaction Amount ($) per Date", table_3a)

        table_3b = self.analytics.fraud_rate_by_hour().copy()
        self.print_ascii_table("3b) Fraud Rate (%) by Hour of Day", table_3b)

        table_3c = self.analytics.fraud_status_distribution()[["label", "count", "percentage"]].copy()
        table_3c = table_3c.rename(columns={"label": "Fraud_Status"})
        self.print_ascii_table("3c) Overall Fraud Status Distribution", table_3c)

        counts, pct = self.analytics.debit_credit_by_fraud_status()
        table_3d = pd.DataFrame({
            "Fraud_Status": counts.index.astype(str),
            "Debit_Count": counts.get("Debit", pd.Series(0, index=counts.index)).astype(int).values,
            "Debit_Percentage": pct.get("Debit", pd.Series(0.0, index=pct.index)).values,
            "Credit_Count": counts.get("Credit", pd.Series(0, index=counts.index)).astype(int).values,
            "Credit_Percentage": pct.get("Credit", pd.Series(0.0, index=pct.index)).values,
        })
        self.print_ascii_table("3d) Debit vs Credit Transactions by Fraud Status", table_3d)

        table_3e = self.analytics.top5_business_types_by_fraud_amount().copy()
        print("\n3e) Top 5 Business Types by Fraudulent Transaction Amount ($)")
        if table_3e.empty:
            self.print_ascii_table("Top 5 Fraudulent Business Types", pd.DataFrame())
        else:
            self.print_ascii_table("Top 5 Fraudulent Business Types", table_3e)


def main():
    
    pipeline = TransactionPipeline(
        transactions_path="transactions.csv",
        merchant_codes_path="merchant_codes.csv",
        fraud_path="fraud.csv"
    )
    pipeline.run()

    analytics = DataAnalytics(pipeline.merged_df)
    report_printer = ReportPrinter(pipeline, analytics)
    report_printer.print_all_reports()

    dashboard = AnalyticsDashboardUI(analytics)
    dashboard.run()


main()