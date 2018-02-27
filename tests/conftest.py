import os

import pytest

from _pytest.capture import CaptureFixture, SysCapture

from lib.utils import config_logging


# Add test environment variable before all tests
def pytest_sessionstart(session):
    config_logging()


def pytest_configure(config):
    """Inject documentation."""
    config.addinivalue_line("markers", "capture_disabled: ")


@pytest.hookimpl(hookwrapper=True)
def pytest_pyfunc_call(pyfuncitem):
    if 'capture_disabled' in pyfuncitem.keywords:
        capmanager = pyfuncitem._request.config.pluginmanager.getplugin('capturemanager')
        capmanager.suspend_capture_item(pyfuncitem._request.node, "call", in_=True)

        print('')
        print('')
        print('======= Capsys Disabled: {} ======= '.format(pyfuncitem.obj.__name__))

    try:
        yield
    finally:
        if 'capture_disabled' in pyfuncitem.keywords:
            print('========= Capsys Enabled ========= ')
            print('')
