import pytest
import numpy as np
import pandas.api.types as ptypes
from pandas.testing import assert_series_equal

from src._user_growth import get_active_user_counts_by_date, get_user_counts_by_date


def test_active_user_count_values_integer():
    dau_count = get_active_user_counts_by_date('2019-01-01', '2020-01-01',
                                               seed=100)
    assert ptypes.is_integer_dtype(dau_count)


def test_user_count_values_integer():
    user_count = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                         seed=100)
    assert ptypes.is_integer_dtype(user_count)


def test_user_count_seed():
    # check same seed gives same vals
    for s in [100, 200, 300, 1000, 5000]:
        user_count_1 = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                               seed=s)
        user_count_2 = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                               seed=s)
        assert_series_equal(user_count_1, user_count_2)

    # check different seeds give different vals
    user_count_1 = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                           seed=100)
    user_count_2 = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                           seed=200)
    assert (user_count_1.sub(user_count_2).max()
            - user_count_1.sub(user_count_2).min() != 0)


def test_user_count_values_increasing():
    # (for defaults)
    user_count = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                         seed=100)
    assert np.all(np.diff(user_count) > 0)


def test_user_count_growth_rate():
    # test approx YoY growth rate is as expected
    # e.g. allow 1dp difference for growth value range

    # this is not exact and will fail for occasional g on some seeds
    start_users = 10000
    for g in [1, 1.5, 2, 2.5, 3, 4]:
        user_count = get_user_counts_by_date('2019-01-01', '2020-01-01',
                                             approx_yoy_growth_rate=g,
                                             start_users=start_users,
                                             seed=100)
        assert np.round(user_count.max() / start_users, decimals=1) == g
