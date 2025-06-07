from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import  OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# TRANSFORMERY GŁÓWNIE ROBIĄ TO SAMO, CO ZROBIONO W 02_FE

# grupowanie po stanach: północ/południe
class StateTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ['address_state']
    def __init__(self):
        self.output_col_ = ["north_state_address"]
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

# grupowanie po celach
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

# grupowanie po zawodach
class JobTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ['emp_title']
    def __init__(self):
        self.output_col_ = ["title_categorized"]
        self.categories = {
            "Healthcare": [r"hospital", r"health", r"kaiser", r"medical", r"clinic", r"pharma", r"nurse", r"doctor",
                           r"healthcare", r"patient"],
            "Retail/Hospitality/Logistics": [r"walmart", r"costco", r"target", r"home depot", r"lowe", r"walgreens", r"macy",
                                   r"starbucks", r"marriott", r"hotel", r"restaurant", r"amazon", r"retail",
                                   r"grocery",r"airline", r"aviation", r"faa", r"airport", r"transport", r"logistics",
                                         r"freight", r"ups", r"fedex", r"truck", r"usps", r"postal service"],
            "Government/Civil Service/Education": [r"federal", r"department of", r"\\bcity of\\b", r"\\bcounty\\b", r"government",
                                         r"homeland", r"irs", r"public", r"state of", r"court", r"postal",
                                         r"air force", r"\\barmy\\b", r"\\bnavy\\b", r"\\bmarine", r"usaf",
                                          r"u\\.s\\.\\s*army",
                                          r"u\\.s\\.\\s*navy", r"u\\.s\\.\\s*air", r"coast guard", r"military",
                                          r"department of defense",
                                          r"\\bdod\\b",r"school", r"university", r"college", r"district", r"teacher", r"professor", r"education",
                          r"institute"]
                                         ,
            "Consulting/Professional Services/Finance/Tech": [r"consult", r"advisor", r"bain", r"mckinsey", r"deloitte", r"pwc",
                                                 r"pricewaterhouse", r"ey", r"kpmg", r"booz allen",
                                                 r"law", r"legal", r"attorney", r"lawyer", r"paralegal",
                                                 r"bank", r"finance", r"financial", r"fidelity", r"tiaa", r"credit",
                                                 r"invest", r"citigroup",
                                                 r"wells", r"chase", r"hartford", r"visa", r"mastercard",
                                                 r"capital [oO]ne", r"goldman",
                                                 r"morgan", r"accountant", r"cpa", r"auditor", r"insurance",
                                                              r"tech", r"software", r"information", r"it", r"comcast",
                                                              r"verizon", r"at[\\s&]*t",
                                                              r"accenture", r"saic", r"northrop", r"lockheed", r"booz",
                                                              r"sprint", r"intel",
                                                              r"microsoft", r"apple", r"google", r"oracle", r"cisco",
                                                              r"ibm", r"hp", r"dell", r"digital",
                                                              r"data"]

            ,
            "Self-Employed": [r"\\bself\\b", r"self[-\\s]?employed", r"freelance", r"owner", r"proprietor",
                              r"entrepreneur"],
            "Manufacturing/Engineering": [r"engineering", r"engineer", r"mfg", r"manufacturing", r"plant",
                                          r"production", r"automotive", r"boeing",r"energy", r"utility", r"power", r"electric", r"gas", r"oil", r"solar"]


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


# transformacja dat
class DateFeaturesTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ["issue_date", "next_payment_date", "last_credit_pull_date", "last_payment_date",
                                                                                "term_months", "loan_status"]
    def __init__(self):
        self.output_col_list_ = ["time_to_last_payment", "daysdiff_issue_pull"]
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
        #df_['pull_after_issue'] = np.where(df_['daysdiff_issue_pull'] > 0, 1, 0)

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
        df_['daysdiff_issue_pull']=(df_['daysdiff_issue_pull']-df_['daysdiff_issue_pull'].mean()) /df_['daysdiff_issue_pull'].std()
        return df_[self.output_col_list_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_list_)


# transformer do progressu w spłatach
class ProgressTransformer(BaseEstimator, TransformerMixin):
    requiredCols=['total_payment', 'installment',
         'term_months', 'loan_amount', 'annual_income']
    def __init__(self):
        self.output_col_ = ['loan_to_income_full_period']

    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all(col in X.columns for col in self.requiredCols ):
            raise ValueError("DataFrame must have all required columns")

        df_ = X.copy()

        # df_['payment_progress'] = df_['total_payment'] / (df_['installment'] * df_['term_months'])
        df_['loan_to_income_full_period'] = df_['loan_amount'] / (df_['annual_income'] * df_['term_months'] / 12)

        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)

# transformer do redukcji skośności    
class SkewnessReductionTransformer(BaseEstimator, TransformerMixin):
    requiredCols = ["installment", 'total_acc', 'total_payment', 'loan_amount']

    def __init__(self):
        self.output_col_ = ["installment_sqrt", 'total_acc_sqrt']

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
        # df_['total_payment_sqrt'] = np.sqrt(df_['total_payment'])
        # df_['loan_amount_sqrt'] = np.sqrt(df_['loan_amount'])

        return df_[self.output_col_].values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_col_)
    
# całościowy pipeline preprocessujący dane
# bez kolumn 'grade_numeric' i 'int_rate', bo silnie skorelowane z grade, a mówią o tym samym
def getPipeline(basic_numeric_cols=['dti', 'annual_income', 'emp_years','sub_grade_numeric'],
                basic_categorical_cols=['home_ownership','loan_status', 'verification_status']):
    # transformer do danych kategorycznych
    ct_onehot = ColumnTransformer([
        ('job', JobTransformer(), JobTransformer.requiredCols),
        ('purpose', PurposeTransformer(), PurposeTransformer.requiredCols),
    ], remainder='drop')

    ct_onehot_cols=JobTransformer.requiredCols+PurposeTransformer.requiredCols

    # transformer do zmiennych z redukowaną skośnością
    ct_scale = ColumnTransformer([
        ('skew', SkewnessReductionTransformer(), SkewnessReductionTransformer.requiredCols)
    ], remainder='drop')

    ct_scale_cols=SkewnessReductionTransformer.requiredCols

    # transformer do dat i podziału na stany
    ct_pass = ColumnTransformer([
        ('date', DateFeaturesTransformer(), DateFeaturesTransformer.requiredCols),
        ('state', StateTransformer(), StateTransformer.requiredCols),
        ('progress', ProgressTransformer(), ProgressTransformer.requiredCols),
        ('total_pass', 'passthrough', [])
    ], remainder='drop')

    ct_pass_cols=np.unique(DateFeaturesTransformer.requiredCols+ StateTransformer.requiredCols+ProgressTransformer.requiredCols).tolist()

    # pipeline'y do cech z FE wraz z onehotem i standaryzacją
    ct1_pipeline = Pipeline([('custom', ct_onehot), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])
    ct2_pipeline = Pipeline([('custom', ct_scale), ('scale', StandardScaler())])
    ct3_pipeline = Pipeline([('custom', ct_pass)])

    # onehot i standaryzacja pierwotnych zmiennych
    basic_num_pipeline = Pipeline([('scale', StandardScaler())])
    basic_cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    # połaczenie wszystkiego w całość
    preprocessor = ColumnTransformer([
        ('ct_onehot', ct1_pipeline, ct_onehot_cols),
        ('ct_scale', ct2_pipeline, ct_scale_cols),
        ('ct_pass', ct3_pipeline, ct_pass_cols),
        ('basic_num', basic_num_pipeline, basic_numeric_cols),
        ('basic_cat', basic_cat_pipeline, basic_categorical_cols),
    ], remainder='drop')

    return preprocessor