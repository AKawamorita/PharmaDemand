import pandas as pd
import numpy as np

class DataQuality:

    def __init__(self, df: pd.DataFrame):
        self.original_df = df.copy()
        self.df = df.copy()
        self.report = None

    CATEGORIES = (
    "M01AB",
    "M01AE",
    "N02BA",
    "N02BE",
    "N05B",
    "N05C",
    "R03",
    "R06"
    )
    CATEGORY_INFO = {
    'M01AB': 'Anti-inflammatory (non-steroid, Acetic acid)',
    'M01AE': 'Anti-inflammatory (non-steroid, Propionic acid)',
    'N02BA': 'Analgesics (Salicylic acid based)',
    'N02BE': 'Analgesics (Pyrazolones/Anilides, e.g. paracetamol-type)',
    'N05B':  'Anxiolytics (anti-anxiety drugs)',
    'N05C':  'Hypnotics/sedatives (sleep drugs)',
    'R03':   'Respiratory drugs (asthma/COPD)',
    'R06':   'Antihistamines (allergy drugs)'
    }

    # =========================================================
    # UTILITÁRIOS
    # =========================================================
    def _safe_numeric(self, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _safe_datetime(self, series: pd.Series) -> pd.Series:
        return pd.to_datetime(series, errors="coerce")

    def _build_report_row(
        self,
        rule_name: str,
        issue_found: bool,
        issue_count: int,
        severity: str,
        recommendation: str
    ) -> dict:
        return {
            "rule_name": rule_name,
            "issue_found": "YES" if issue_found else "NO",
            "issue_count": int(issue_count),
            "severity": severity,
            "recommendation": recommendation
        }

    # =========================================================
    # ANÁLISE
    # =========================================================
    def analyze_data_missing(self,colum_name:str) -> dict:
        qty = self._safe_numeric(self.df[colum_name])
        mask = qty.isna()
        return self._build_report_row(
            rule_name="Missing Quantity",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Medium",
            recommendation="Impute missing {colum_name} with zero if business meaning is 'no units sold'."
        )

    def analyze_data_negative(self,colum_name:str) -> dict:
        qty = self._safe_numeric(self.df[colum_name])
        mask = qty < 0
        return self._build_report_row(
            rule_name="Negative Quantity",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="High",
            recommendation="Replace negative {colum_name} with zero or review source transactions."
        )    

    def analyze_customer_type_invalid(self,colum_name:str,valid_types: set[str]) -> dict:
        mask = ~self.df["Customer_Type"].isin(valid_types)
        return self._build_report_row(
            rule_name="Invalid Customer Type",
            issue_found=mask.any(),
            issue_count=mask.sum(),
            severity="Medium",
            recommendation="Review or standardize {colum_name} values."
        )

    # estas validacoes elas sao pertinentes a um projeto anterior e nao servem para este por isso estao comentados
    #
    # def analyze_order_date_invalid(self,colum_name:str) -> dict:
    #     dates = self._safe_datetime(self.df["colum_name"])
    #     mask = dates.isna()
    #     return self._build_report_row(
    #         rule_name="Invalid Order Date",
    #         issue_found=mask.any(),
    #         issue_count=mask.sum(),
    #         severity="High",
    #         recommendation="Parse {colum_name} and review invalid date records."
    #     )

    # def analyze_duplicates(self) -> dict:
    #     mask = self.df.duplicated()
    #     return self._build_report_row(
    #         rule_name="Exact Duplicate Rows",
    #         issue_found=mask.any(),
    #         issue_count=mask.sum(),
    #         severity="Medium",
    #         recommendation="Remove exact duplicate rows."
    #     )

    # def analyze_total_price_consistency(self, tolerance: float = 0.1) -> dict:
    #     unit_price = self._safe_numeric(self.df["Unit_Price"])
    #     qty = self._safe_numeric(self.df["Quantity"])
    #     disc = self._safe_numeric(self.df["Discount"]).fillna(0).clip(0, 1)
    #     total_price = self._safe_numeric(self.df["Total_Price"]).round(2)

    #     expected_total = (unit_price * qty * (1 - disc)).round(2)
    #     diff = (expected_total - total_price).abs().round(2)

    #     mask = (
    #         unit_price.notna() &
    #         qty.notna() &
    #         total_price.notna() &
    #         (diff > tolerance)
    #     )

    #     return self._build_report_row(
    #         rule_name="Total Price Consistency",
    #         issue_found=bool(mask.any()),
    #         issue_count=int(mask.sum()),
    #         severity="High",
    #         recommendation="Recalculate Total_Price."
    #     )

    #     "M01AB",
#     "",
#     "",
#     "",
#     "",
#     "",
#     "",
#     ""

    def run_full_analysis(self, tolerance: float = 0.01) -> pd.DataFrame:
        rows = [
            self.analyze_quantity_missing("M01AB"),
            self.analyze_quantity_negative("M01AB"),
            self.analyze_quantity_missing("M01AE"),
            self.analyze_quantity_negative("M01AE"),
            self.analyze_quantity_missing("N02BA"),
            self.analyze_quantity_negative("N02BA"),
            self.analyze_quantity_missing("N02BE"),
            self.analyze_quantity_negative("N02BE"),
            self.analyze_quantity_missing("N05B"),
            self.analyze_quantity_negative("N05B"),
            self.analyze_quantity_missing("N05C"),
            self.analyze_quantity_negative("N05C"),
            self.analyze_quantity_missing("R03"),
            self.analyze_quantity_negative("R03"),
            self.analyze_quantity_missing("R06"),
            self.analyze_quantity_negative("R06"),
            #self.analyze_duplicates(),
            #self.analyze_total_price_consistency(tolerance=tolerance)
        ]
        self.report = self.report = pd.DataFrame(rows).drop(columns=["recommendation"], errors="ignore")
        return self.report
    
    # =========================================================
    # CORREÇÕES
    # =========================================================
    def fix_quantity_missing_with_zero(self,colum_name:str):
        self.df["Quantity"] = self._safe_numeric(self.df[colum_name]).fillna(0)
        return self.df

    def fix_quantity_negative_with_zero(self,colum_name:str):
        self.df["Quantity"] = self._safe_numeric(self.df[colum_name])
        self.df.loc[self.df["Quantity"] < 0, "Quantity"] = 0
        return self.df

    def fix_discount_missing_with_zero(self,colum_name:str):
        self.df["Discount"] = self._safe_numeric(self.df[colum_name]).fillna(0)
        return self.df
    
    def fix_sales_value_by_neighbor_mean(
        self,
        colum_name:str,
        group_cols=("Product", "Region"),
        target_col="Total_Price"        
    ):
        """
        Corrige valores ausentes em target_col usando média entre d-1 e d+1 dentro do grupo.
        Requer ordenação temporal por uma coluna (data) informada.
        """
        self.df[colum_name] = self._safe_datetime(self.df[colum_name])
        self.df[target_col] = self._safe_numeric(self.df[target_col])

        self.df = self.df.sort_values(list(group_cols) + [colum_name]).copy()

        def _fill_group(s):
            prev_val = s.shift(1)
            next_val = s.shift(-1)
            neighbor_mean = (prev_val + next_val) / 2
            return s.fillna(neighbor_mean)

        self.df[target_col] = (
            self.df.groupby(list(group_cols))[target_col]
            .transform(_fill_group)
        )

        return self.df

    # Nao aplicado a este projeto
    # def apply_all_fixes(self, recalc_total_price: bool = True):
    #     self.fix_quantity_missing_with_zero()
    #     self.fix_quantity_negative_with_zero()
    #     self.fix_discount_missing_with_zero()
    #     self.fix_discount_negative_with_zero()
    #     self.fix_discount_clip_range()
    #     self.fix_b2c_discount_rule()
    #     self.fix_order_date_parse()
    #     self.fix_remove_duplicates()

    #     if recalc_total_price:
    #         self.fix_total_price_recalculate()

    #     return self.df

    # =========================================================
    # RELATÓRIO PROFISSIONAL
    # =========================================================
    def generate_executive_summary(self) -> str:
        if self.report is None:
            self.run_full_analysis()

        total_rules = len(self.report)
        issues_found = (self.report["issue_found"] == "YES").sum()
        total_issues = self.report["issue_count"].sum()

        high_issues = self.report[
            (self.report["issue_found"] == "YES") &
            (self.report["severity"] == "High")
        ]["rule_name"].tolist()

        if high_issues:
            high_issues_text = ", ".join(high_issues)
        else:
            high_issues_text = "No high-severity issues were detected"

        summary = (
            f"Data Quality Executive Summary\n"
            f"------------------------------\n"
            f"Rules evaluated: {total_rules}\n"
            f"Rules with issues found: {issues_found}\n"
            f"Total issue count: {total_issues}\n"
            f"High-severity issues: {high_issues_text}\n"
        )
        return summary

    def get_clean_dataframe(self) -> pd.DataFrame:
        return self.df.copy()

    def reset_to_original(self):
        self.df = self.original_df.copy()
        self.report = None

    @staticmethod
    def sales_series_from_dataframe(df_salesdaily: pd.DataFrame, category: str, target_col: str) -> pd.Series:
        """Usa df_salesdaily em memória. Não imputa dias nem quantidades ausentes."""
        if category not in DataQuality.CATEGORIES:
            raise ValueError(f"Categoria {category!r} inválida. Opções: {DataQuality.CATEGORIES}")
        missing = set(["datum", category]) - set(df_salesdaily.columns)

        if missing:
            raise ValueError(f"Colunas ausentes em df_salesdaily: {sorted(missing)}")
        df = df_salesdaily.loc[:, ["datum", category]].copy()

        if not pd.api.types.is_datetime64_any_dtype(df["datum"]):
            df["datum"] = pd.to_datetime(df["datum"], format="%m/%d/%Y", errors="raise")
        df[category] = pd.to_numeric(df[category], errors="raise")

        if df["datum"].duplicated().any():
            raise ValueError("Datas duplicadas em df_salesdaily")
        df = df.sort_values("datum").set_index("datum")

        if df[category].isna().any() or df[category].lt(0).any():
            raise ValueError("Vendas ausentes/negativas: corrija a fonte antes de criar features")
        expected = pd.date_range(df.index.min(), df.index.max(), freq="D")
        
        if not df.index.equals(expected):
            raise ValueError("Datas ausentes: não serão imputadas sem regra de negócio")
        return df[category].astype(float).rename(target_col)