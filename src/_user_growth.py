import pandas as pd
import numpy as np


def _get_active_user_p_by_date(start_date, end_date, *, seed=None):
    """Generate a trajectory for % daily active users (DAU).
    The series is a random walk with drift.

    DAU % starts between approx. 20-30% range and then follows random walk
    with drift over time.  In bad cases, DAU % might halve within a year.

    Args:
        start_date (str): Y-m-d date string for the start of the series.
        end_date (str): Y-m-d date string for the end of the series.
        approx_yoy_growth_rate (int, optional): YoY growth rate for the
            user count (>=1), e.g. 2 for +100%, 3 for +200%. Defaults to 3.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        [type]: [description]
    """
    if seed:
        np.random.seed(seed)

    date_index = pd.date_range(start_date, end_date,
                               closed='left')

    # set up random walk (steps cumsum for drift)
    ACTIVE_P = np.random.normal(0.25, 0.03)
    steps = np.random.normal(0, 0.002, size=len(date_index))

    DAU_p_draw = ((np.zeros(len(date_index))
                   + ACTIVE_P + steps.cumsum())
                  .clip(0, 1))
    return pd.Series(DAU_p_draw, index=date_index)


def get_user_counts_by_date(start_date, end_date, *,
                            approx_yoy_growth_rate=3, start_users=10000,
                            seed=None):
    """Get count of number of users of the product/business, indexed by
    date, given these parameters:
    - Date range
    - approx_yoy_growth_rate of users (people who have signed up in total)
    - start_users (number of users on start date)

    If using default arguments and a year range, the number of users
    would go from 10k to approx. 30k (growth_rate=3 -> +200%) by the end
    of the series.

    Args:
        start_date (str): Y-m-d date string for the start of the series.
        end_date (str): Y-m-d date string for the end of the series.
        approx_yoy_growth_rate (int, optional): YoY growth rate for the
            user count (>=1), e.g. 2 for +100%, 3 for +200%. Defaults to 3.
        start_users (int, optional): Number of users at the start date.
            Defaults to 10000.
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        pd.Series
    """
    date_index = pd.date_range(start_date, end_date,
                               closed='left')

    if seed:
        np.random.seed(seed)

    dod_growth_rate = approx_yoy_growth_rate ** (1/365)

    # set up random walk (no drift)
    steps = np.random.normal(dod_growth_rate, 0.0005,
                             size=len(date_index))

    n_users_draw = ((np.zeros(len(date_index)) + start_users) *
                    steps.cumprod()).astype(int)
    return pd.Series(n_users_draw, index=date_index)


def get_active_user_counts_by_date(start_date, end_date, *,
                                   approx_yoy_growth_rate=3, start_users=10000,
                                   seed=None):
    """Get count of number of daily active users (DAU) of the product/service,
    indexed by date, given these parameters:
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
        pd.Series
    """
    active_user_p_by_date = _get_active_user_p_by_date(start_date, end_date,
                                                       seed=seed)
    users_by_date = get_user_counts_by_date(start_date, end_date,
                                            seed=seed,
                                            approx_yoy_growth_rate=approx_yoy_growth_rate,
                                            start_users=start_users)

    return (users_by_date * active_user_p_by_date).astype(int)
