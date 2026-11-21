# Copyright (c) ModelScope Contributors. All rights reserved.
from types import SimpleNamespace

import pytest

from swift.dev.recipe.utils import close_driver_preserving_error


def test_driver_close_preserves_primary_error_and_continues_cleanup():
    events = []
    primary = ValueError('training failed')
    close_error = BaseExceptionGroup('close failed', [RuntimeError('episode stop failed')])

    def close():
        events.append('driver.close')
        raise close_error

    driver = SimpleNamespace(close=close)
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        finally:
            close_driver_preserving_error(driver)
            events.append('reclaim')

    assert caught.value is primary
    assert events == ['driver.close', 'reclaim']


def test_driver_close_error_propagates_without_primary_error():
    close_error = TimeoutError('drain timed out')
    driver = SimpleNamespace(close=lambda: (_ for _ in ()).throw(close_error))

    with pytest.raises(TimeoutError) as caught:
        close_driver_preserving_error(driver)

    assert caught.value is close_error


def test_driver_close_success_returns_normally():
    events = []
    close_driver_preserving_error(SimpleNamespace(close=lambda: events.append('driver.close')))

    assert events == ['driver.close']
