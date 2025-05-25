from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class StateTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ['address_state']
    def __init__(self):
        self.output_col_ = ["north_state_address", "address_state"]
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
        if not any(col in X.columns for col in self.requiredCols):
            raise ValueError("DataFrame must contain required columns")
        unknown = ~X["address_state"].isin(self.North | self.South)
        if unknown.any():
            bad = X.loc[unknown, "address_state"].unique()
            raise ValueError(f"Unknown states encountered: {bad.tolist()}")
        df_ = X.copy()
        df_['north_state_address'] = df_['address_state'].apply(lambda x: 1 if x in self.North else 0)
        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)

class PurposeTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ['purpose']
    def __init__(self):
        self.output_col_ = ["purpose_clustered"]
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
        if not any(col in X.columns for col in self.requiredCols):
            raise ValueError("DataFrame must have an 'purpose' column")

        df_ = X.copy()

        df_['purpose_clustered'] = df_['purpose'].apply(self._purpose_clustering)
        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)


class JobTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ['emp_title']
    def __init__(self):
        self.output_col_ = ["title_categorized"]
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
        if not any(col in X.columns for col in self.requiredCols):
            raise ValueError("DataFrame must have an 'purpose' column")
        df_ = X.copy()
        df_["title_categorized"] = df_["emp_title"].apply(self._classify)
        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)


class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ["issue_date", "next_payment_date", "last_credit_pull_date", "last_payment_date",
                                                                                "term_months", "loan_status"]
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


        if not any(col in X.columns for col in self.requiredCols):
            raise ValueError("DataFrame must contain required columns")

        df_ = X.copy()

        df_['next_payment_date'] = pd.to_datetime(df_['next_payment_date'], format='%d-%m-%Y')
        df_['last_payment_date'] = pd.to_datetime(df_['last_payment_date'], format='%d-%m-%Y')
        df_['issue_date'] = pd.to_datetime(df_['issue_date'], format='%d-%m-%Y')
        df_['last_credit_pull_date'] = pd.to_datetime(df_['last_credit_pull_date'], format='%d-%m-%Y')

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
        df_['time_to_last_payment']=(df_['time_to_last_payment'] - df_['time_to_last_payment'].mean()) / df_['time_to_last_payment'].std()
        return df_[self.output_col_list_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_list_)

class AutoFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self._ct = ColumnTransformer(transformers=[])

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())

        cols = ['state__north_state_address',
                'state__address_state',
                'purpose__purpose_clustered',
                'job__title_categorized',
                'date__time_to_last_payment',
                'date__pull_after_issue',
                'date__term_months',
                'date__loan_status',
                'progress__payment_progress',
                'progress__loan_to_income',
                'progress__annual_income',
                'skewness__installment_sqrt',
                'skewness__total_acc_sqrt',
                'skewness__total_payment_sqrt',
                'remainder__home_ownership',
                'remainder__verification_status',
                'remainder__dti',
                'remainder__int_rate',
                'remainder__emp_years',
                'remainder__grade_numeric',
                'remainder__sub_grade_numeric',
                'remainder__issue_year',
                'remainder__issue_month',
                'remainder__issue_day',
                'remainder__issue_weekday',
                'remainder__last_credit_pull_year',
                'remainder__last_credit_pull_month',
                'remainder__last_credit_pull_day',
                'remainder__last_credit_pull_weekday',
                'remainder__last_payment_year',
                'remainder__last_payment_month',
                'remainder__last_payment_day',
                'remainder__last_payment_weekday',
                'remainder__next_payment_year',
                'remainder__next_payment_month',
                'remainder__next_payment_day',
                'remainder__next_payment_weekday']

        binary_cols = ['state__north_state_address','date__time_to_last_payment','date__pull_after_issue']
        categorical_cols = ['state__address_state','purpose__purpose_clustered','job__title_categorized', 'date__term_months', 'date__loan_status',
                            'remainder__home_ownership','remainder__verification_status', 'remainder__emp_years']
        numeric_cols = [col for col in cols if col not in binary_cols + categorical_cols]

        self._ct = ColumnTransformer(
            transformers=[
                ("bin",     "passthrough",  binary_cols),
                ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
                ("scale",   StandardScaler(copy=True),            numeric_cols)
            ],
            remainder="drop",
        )

        self._ct.fit(df)
        return self

    def transform(self, X):
        return self._ct.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self._ct.get_feature_names_out(input_features)

def getPipeline(basic_numeric_cols,basic_categorical_cols):
    ct_onehot = ColumnTransformer([
        ('job', JobTransformer(), JobTransformer.requiredCols),
        ('purpose', PurposeTransformer(), PurposeTransformer.requiredCols),
    ], remainder='drop')

    ct_onehot_cols=JobTransformer.requiredCols+PurposeTransformer.requiredCols

    ct_scale = ColumnTransformer([
        ('skew', SkewnessReductionTransformer(), SkewnessReductionTransformer.requiredCols)
    ], remainder='drop')

    ct_scale_cols=SkewnessReductionTransformer.requiredCols

    ct_pass = ColumnTransformer([
        ('date', DateFeaturesTransformer(), DateFeaturesTransformer.requiredCols),
        ('state', StateTransformer(), StateTransformer.requiredCols)
    ], remainder='drop')

    ct_pass_cols=DateFeaturesTransformer.requiredCols+ StateTransformer.requiredCols
    ct1_pipeline = Pipeline([('custom', ct_onehot), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    ct2_pipeline = Pipeline([('custom', ct_scale), ('scale', StandardScaler())])
    ct3_pipeline = Pipeline([('custom', ct_pass)])

    basic_num_pipeline = Pipeline([('scale', StandardScaler())])
    basic_cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('ct_onehot', ct1_pipeline, ct_onehot_cols),
        ('ct_scale', ct2_pipeline, ct_scale_cols),
        ('ct_state', ct3_pipeline, ct_pass_cols),
        ('basic_num', basic_num_pipeline, basic_numeric_cols),
        ('basic_cat', basic_cat_pipeline, basic_categorical_cols),
    ], remainder='drop')

    return preprocessor

class ProgressTransformer(BaseEstimator, TransformerMixin):
    requiredCols=['total_payment', 'installment',
         'term_months', 'loan_amount', 'annual_income']
    def __init__(self):
        self.output_col_ = ["payment_progress", 'loan_to_income', 'annual_income']

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all(col in X.columns for col in self.requiredCols ):
            raise ValueError("DataFrame must have all required columns")

        df_ = X.copy()

        # Czy to dziala dla tych co splacili? te dane troche kret ja bym sprawdzil
        df_['payment_progress'] = df_['total_payment'] / (df_['installment'] * df_['term_months'])

        df_['loan_to_income'] = df_['loan_amount'] / df_['annual_income']  # To juz ma dti w jakims stopniu
        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)
    
class SkewnessReductionTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ["installment", 'total_acc', 'total_payment']

    def __init__(self):
        self.output_col_ = ["installment_sqrt", 'total_acc_sqrt', 'total_payment_sqrt']

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all(col in X.columns for col in self.requiredCols):
            raise ValueError("DataFrame must have all required columns")

        df_ = X.copy()

        df_['installment_sqrt'] = np.sqrt(df_['installment'])
        df_['total_acc_sqrt'] = np.sqrt(df_['total_acc'])
        df_['total_payment_sqrt'] = np.sqrt(df_['total_payment'])
        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)