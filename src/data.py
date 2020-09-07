import uuid
import pandas as pd
import numpy as np
from scipy.stats import skewnorm

from src._user_growth import get_user_counts_by_date, get_active_user_counts_by_date


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
    user_df['user_id'] = _get_uuid_values(total_users)  # not seeded
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


def _get_uuid_values(n):
    ids_list = [str(uuid.uuid4())
                for i in range(n)]
    return ids_list


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


def get_user_activity_dataset(start_date, end_date, *, users_df,
                              approx_yoy_growth_rate=3, start_users=10000,
                              seed=None):
    """Get dataset of user activity information, e.g. session timestamps.
    Pass in the user dataset as users_df and the arguments used to generate
    that dataset (including the seed so that user counts align over time
    for the data generation to work).

    Given these parameters used to simulate the SaaS business:
        - Date range
        - approx_yoy_growth_rate of users (people who have signed up in total)
        - start_users (number of users on start date)

    With user activity dataset returned and users_df joined together, you
    can calculate interesting product metrics such as active users over time,
    user retention over time, etc.

    Args:
        start_date (str): Y-m-d date string for the start of the series.
        end_date (str): Y-m-d date string for the end of the series.
        users_df (pd.DataFrame): user data to utilise in the data
            generation process for this 'activity' dataset.
        approx_yoy_growth_rate (int, optional): YoY growth rate for the
            user count (>=1), e.g. 2 for +100%, 3 for +200%. Defaults to 3.
        start_users (int, optional): Number of users at the start date.
            Defaults to 10000.
        seed (int): Random seed.  This should be the same as the seed used
            to generate users_df.

    Returns:
        pd.DataFrame
    """
    active_user_counts = get_active_user_counts_by_date(
        start_date, end_date,
        approx_yoy_growth_rate=approx_yoy_growth_rate,
        start_users=start_users,
        seed=seed)

    # define initial col order
    columns = ['session_id', 'user_id', 'session_start_date']

    activity_df = pd.DataFrame(columns=columns)
    activity_df['session_start_date'] = pd.Series((active_user_counts.dropna()
                                                   .repeat(active_user_counts.dropna()))
                                                  .index.values)
    activity_df['session_id'] = _get_uuid_values(len(activity_df))  # not seeded
    # choose active user IDs via a sampling function
    activity_df = (activity_df
                   .pipe(_sample_user_ids,
                         users_df=users_df,
                         active_user_counts=active_user_counts,
                         start_date=start_date,
                         end_date=end_date,
                         seed=seed)
                   .drop(columns=['user_activation_date'])  # drop joined col
                   )

    return activity_df.set_index('session_id')


def _sample_user_ids(activity_df, *, users_df, active_user_counts,
                     start_date, end_date, seed=None):
    df = activity_df.copy()

    # on avg, uuid digits sum near 100 and give norm dist
    sum_of_user_id_digits = [sum(i)
                             for i in [[int(i) for i in ''.join(s)]
                                       for s in (users_df.index
                                                 .str.findall("(\d*\.?\d+)"))]]
    # make stickiness score from UUID (positive skew)
    stickiness_score = (pd.Series(sum_of_user_id_digits,
                                  index=users_df.index) ** 2 // 100)

    for d in pd.date_range(start_date, end_date, closed='left'):
        list_active_users_to_add = []

        dau = active_user_counts.loc[d]

        filtered_df = users_df[users_df['activation_date'] <= d].copy()
        filtered_df['days_since_activation'] = (d - filtered_df['activation_date']).dt.days

        new_users = np.unique(filtered_df[filtered_df['days_since_activation'] == 0]
                              .index.tolist())

        # make all new users active for this day
        list_active_users_to_add.extend(new_users)

        # for remaining active users to fill, use a sampling function
        active_users_left_to_add = dau - len(new_users)

        active_existing_user_ids = _draw_existing_users_to_set_as_active(
            filtered_df,
            stickiness_score=stickiness_score,
            n_draws=active_users_left_to_add
        )

        list_active_users_to_add.extend(active_existing_user_ids)

        # set user ids
        df.loc[df['session_start_date'] == d,
               'user_id'] = list_active_users_to_add

    return (df
            .merge(users_df['activation_date'].rename('user_activation_date'),
                   how='left', left_on=['user_id'], right_index=True))


def _draw_existing_users_to_set_as_active(existing_users_df, *, stickiness_score, n_draws):
    filtered_df = existing_users_df[existing_users_df['days_since_activation'] > 0].copy()
    existing_users = np.unique(filtered_df.index.tolist())

    # get weights for users who were created before date (inclusive) (sum to 1)
    stickiness_weights = stickiness_score.filter(existing_users, axis=0)

    # decay function (horizontal asymptote to encourage an engagement floor)
    filtered_df['stickiness_decay_factor'] = 0.8 * 0.9 ** filtered_df['days_since_activation'] + 0.1

    # weighted function (of sticky user IDs and of time since activation )
    stickiness_weights = 0.2 * stickiness_weights + 0.8 * (40 * filtered_df['stickiness_decay_factor'])
    # sum weights to 1 before doing the sampling
    stickiness_weights = stickiness_weights / stickiness_weights.sum()

    # sample active users via stickiness weights (sample -> no replacement)
    active_existing_user_ids = np.random.choice(stickiness_weights.index.tolist(),
                                                size=n_draws,
                                                replace=False,
                                                p=stickiness_weights.tolist())
    return active_existing_user_ids
