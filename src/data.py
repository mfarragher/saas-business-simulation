import uuid
import pandas as pd
import numpy as np
from scipy.stats import skewnorm

from src._user_growth import get_user_counts_by_date


def get_user_dataset(start_date, end_date, *,
                     approx_yoy_growth_rate=3, start_users=10000,
                     seed=None):
    """Get dataset of user account information, e.g. activation date,
    age, country.
    Given these parameters used to simulate the SaaS business:
    - Date range
    - approx_yoy_growth_rate of users (people who have signed up in total)
    - start_users (number of users on start date)

    Args:
        start_date (str): Y-m-d date string for the start of the series.
        end_date (str): Y-m-d date string for the end of the series.
        approx_yoy_growth_rate (int, optional): YoY growth rate for the
            user count (>=1), e.g. 2 for +100%, 3 for +200%. Defaults to 3.
        start_users (int, optional): Number of users at the start date.
            Defaults to 10000.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        pd.DataFrame
    """
    columns = ['user_id', 'activation_date', 'country', 'age']

    new_users_by_date = _get_new_user_counts_by_date(start_date, end_date,
                                                     seed=seed,
                                                     approx_yoy_growth_rate=approx_yoy_growth_rate,
                                                     start_users=start_users)

    new_users_over_time = new_users_by_date.sum()
    total_users = int(start_users + new_users_over_time)

    user_df = pd.DataFrame(columns=columns)
    user_df['user_id'] = _get_user_id_values(total_users)  # not seeded
    user_df['age'] = _get_age_dist_values(total_users, seed=seed)
    user_df['country'] = _get_user_country_values(total_users, seed=seed)
    user_df = user_df.pipe(_fill_activation_dates,
                           new_users_by_date=new_users_by_date,
                           start_users=start_users)
    return user_df.set_index('user_id')


def _get_new_user_counts_by_date(start_date, end_date, *,
                                 approx_yoy_growth_rate=3, start_users=10000,
                                 seed=None):
    users_by_date = get_user_counts_by_date(start_date, end_date,
                                            seed=seed,
                                            approx_yoy_growth_rate=approx_yoy_growth_rate,
                                            start_users=start_users)
    return users_by_date - users_by_date.shift(1)


def _get_user_id_values(total_users):
    user_ids = [str(uuid.uuid4())
                for i in range(total_users)]
    return user_ids


def _get_age_dist_values(total_users, *, seed=None):
    """Return Numpy array of age values, given total_users"""
    if seed:
        np.random.seed(seed)

    # negative skewnorm dist - age range of 16 to 60 ish
    return ((27 + 10 * skewnorm.rvs(a=2, size=total_users))
            .astype(int)
            .clip(min=16))


def _get_user_country_values(total_users, *, seed=None):
    if seed:
        np.random.seed(seed)

    num_vals = np.random.choice([0, 1], p=[0.2, 0.8],
                                size=total_users)
    return np.where(num_vals == 1, 'US', 'CA')


def _fill_activation_dates(df, *, new_users_by_date, start_users, seed=None):
    # legacy users - assume dates uniformly distributed in year before:
    legacy_end_date = new_users_by_date.index.min()
    legacy_start_date = legacy_end_date - pd.DateOffset(years=1)
    legacy_user_date_range = pd.date_range(legacy_start_date, legacy_end_date,
                                           closed='left').to_series()

    legacy_dates = (legacy_user_date_range
                    .sample(start_users,
                            replace=True,
                            random_state=seed)
                    .reset_index(drop=True))

    # repeat the dates based on value counts
    new_dates = pd.Series((new_users_by_date.dropna()
                           .repeat(new_users_by_date.dropna()))
                          .index.values)
    # concat the legacy & new activation dates
    date_values = (pd.concat([legacy_dates, new_dates],
                             ignore_index=True).values)
    return df.assign(activation_date=lambda x: date_values)
