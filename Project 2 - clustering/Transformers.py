from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
from pandas.tseries.offsets import DateOffset
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion

class StateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_col_ = "north_state_address"
        self.North = {'CO', 'CT', 'DE', 'DC','ID', 'IL', 'IA', 'IN', 'ME', 'MD', 'MA',
                      'MI', 'MN', 'MT', 'NE', 'NH', 'NJ', 'ND', 'NV', 'NY', 'OH',
                      'OR', 'PA', 'RI', 'SD', 'UT', 'VT', 'WA', 'WI', 'WY'}

        self.South = {'AL', 'AK', 'AZ', 'AR', 'CA', 'FL', 'GA', 'HI', 'KS', 'KY',
                      'LA', 'MO', 'MS', 'NC', 'NM', 'OK', 'SC', 'TN', 'TX', 'VA', 'WV'}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if "address_state" not in X.columns:
            raise ValueError("DataFrame must have an 'address_state' column")
        unknown = ~X["address_state"].isin(self.North | self.South)
        if unknown.any():
            bad = X.loc[unknown, "address_state"].unique()
            raise ValueError(f"Unknown states encountered: {bad.tolist()}")
        df_ = X.copy()
        df_['north_south_address'] = df_['address_state'].apply(lambda x: 1 if x in self.North else 0)
        return df_[['north_south_address']].values

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_col_])


class PurposeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_col_ = "purpose_clustered"
        self.life = {'car', 'home', 'home improvement', 'house', 'medical', 'moving', 'vacation', 'wedding',
                     'major purchase',
                     'renewable_energ'}
        self.finance = {'credit card', 'Debt consolidation', 'small business'}

    def fit(self, X, y=None):
        return self

    def _purpose_clustering(self, x):
        if x in self.life:
            return "life"
        elif x in self.finance:
            return "finance"
        else:
            return "other"

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if "purpose" not in X.columns:
            raise ValueError("DataFrame must have an 'purpose' column")

        df_ = X.copy()

        df_['purpose_clustered'] = df_['purpose'].apply(self._purpose_clustering)
        return df_[['purpose_clustered']].values

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_col_])


class JobTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_col_ = "title_categorized"
        self.categories = {
            "Tech/Telecom": [r"tech", r"software", r"information", r"it", r"comcast", r"verizon", r"at[\\s&]*t",
                             r"accenture", r"saic", r"northrop", r"lockheed", r"booz", r"sprint", r"intel",
                             r"microsoft", r"apple", r"google", r"oracle", r"cisco", r"ibm", r"hp", r"dell", r"digital",
                             r"data"],
            "Finance": [r"bank", r"finance", r"financial", r"fidelity", r"tiaa", r"credit", r"invest", r"citigroup",
                        r"wells", r"chase", r"hartford", r"visa", r"mastercard", r"capital [oO]ne", r"goldman",
                        r"morgan", r"accountant", r"cpa", r"auditor", r"insurance"],
            "Military": [r"air force", r"\\barmy\\b", r"\\bnavy\\b", r"\\bmarine", r"usaf", r"u\\.s\\.\\s*army",
                         r"u\\.s\\.\\s*navy", r"u\\.s\\.\\s*air", r"coast guard", r"military", r"department of defense",
                         r"\\bdod\\b"],
            "Education": [r"school", r"university", r"college", r"district", r"teacher", r"professor", r"education",
                          r"institute"],
            "Healthcare": [r"hospital", r"health", r"kaiser", r"medical", r"clinic", r"pharma", r"nurse", r"doctor",
                           r"healthcare", r"patient"],
            "Retail/Hospitality": [r"walmart", r"costco", r"target", r"home depot", r"lowe", r"walgreens", r"macy",
                                   r"starbucks", r"marriott", r"hotel", r"restaurant", r"amazon", r"retail",
                                   r"grocery"],
            "Transportation/Logistics": [r"airline", r"aviation", r"faa", r"airport", r"transport", r"logistics",
                                         r"freight", r"ups", r"fedex", r"truck", r"usps", r"postal service"],
            "Government/Civil Service": [r"federal", r"department of", r"\\bcity of\\b", r"\\bcounty\\b", r"government",
                                         r"homeland", r"irs", r"public", r"state of", r"court", r"postal"],
            "Consulting/Professional Services": [r"consult", r"advisor", r"bain", r"mckinsey", r"deloitte", r"pwc",
                                                 r"pricewaterhouse", r"ey", r"kpmg", r"booz allen"],
            "Self-Employed": [r"\\bself\\b", r"self[-\\s]?employed", r"freelance", r"owner", r"proprietor",
                              r"entrepreneur"],
            "Manufacturing/Engineering": [r"engineering", r"engineer", r"mfg", r"manufacturing", r"plant",
                                          r"production", r"automotive", r"boeing"],
            "Energy/Utilities": [r"energy", r"utility", r"power", r"electric", r"gas", r"oil", r"solar"],
            "Legal/Law": [r"law", r"legal", r"attorney", r"lawyer", r"paralegal"],
        }

    def fit(self, X, y=None):
        return self

    def _classify(self, title):
        t = title.lower()
        for cat, patterns in self.categories.items():
            for p in patterns:
                if re.search(p, t):
                    return cat
        return "Other"

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if "emp_title" not in X.columns:
            raise ValueError("DataFrame must have an 'purpose' column")
        df_ = X.copy()
        df_["title_categorized"] = df_["emp_title"].apply(self._classify)
        return df_[['title_categorized']].values

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_col_])


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.output_col_list_ = ["time_to_last_payment", "pull_after_issue"]

    def fit(self, X, y=None):
        return self

    def _date_fixer(self, x, y):
        if x == 'Fully Paid':
            return 0
        else:
            return y

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        required = ["issue_date", "next_payment_date", "last_credit_pull_date", "last_payment_date"
                                                                                "term_months"]
        if not any(col in X.columns for col in required):
            raise ValueError("DataFrame must contain required columns")

        df_ = X.copy()

        df_['next_payment_date'] = pd.to_datetime(df_['next_payment_date'], format='%Y-%m-%d')
        df_['last_payment_date'] = pd.to_datetime(df_['last_payment_date'], format='%Y-%m-%d')
        df_['issue_date'] = pd.to_datetime(df_['issue_date'], format='%Y-%m-%d')
        df_['last_credit_pull_date'] = pd.to_datetime(df_['last_credit_pull_date'], format='%Y-%m-%d')

        df_['daysdiff_issue_pull'] = (df_['issue_date'] - df_['last_credit_pull_date']).dt.days
        df_['pull_after_issue'] = np.where(df_['daysdiff_issue_pull'] > 0, 1, 0)

        df_['end_date'] = df_.apply(
            lambda row: row['issue_date'] + DateOffset(months=row['term_months']),
            axis=1
        )
        df_['time_to_last_payment'] = (df_['end_date'] - df_['next_payment_date']).dt.days

        df_['time_to_last_payment'] = df_.apply(
            lambda row: self._date_fixer(row['loan_status'],
                                         row['time_to_last_payment']),
            axis=1
        )

        return df_[["pull_after_issue", "time_to_last_payment"]].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_list_)

class AutoFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._ct = None

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())

        binary_cols = [i for i in df.columns if set(df[i].dropna().unique()) <= {0, 1}]
        numeric_cols = [i for i in df.select_dtypes(include=[np.number]).columns
                        if i not in binary_cols]
        categorical_cols = [i for i in df.columns if i not in binary_cols + numeric_cols]

        self._ct = ColumnTransformer(
            transformers=[
                ("bin",     "passthrough",               binary_cols),
                ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ("scale",   StandardScaler(),            numeric_cols),
            ],
            remainder="drop",
        )

        self._ct.fit(df)
        return self

    def transform(self, X):
        return self._ct.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self._ct.get_feature_names_out(input_features)


class MergeFeaturesDF(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns, transformers):
        self.keep_columns  = keep_columns
        self.transformers  = transformers

    def fit(self, X, y=None):
        for _, tr in self.transformers:
            tr.fit(X, y)
        return self

    def transform(self, X):

        df = X[self.keep_columns].copy()

        for name, tr in self.transformers:
            cols   = tr.get_feature_names_out(None)
            values = tr.transform(X)
            df_tr  = pd.DataFrame(values, columns=cols, index=df.index)
            df     = pd.concat([df, df_tr], axis=1)

        return df
