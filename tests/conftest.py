"""Shared pytest fixtures for all OvoScan tests."""

import os
import pytest


@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_LOG_LEVEL", "WARNING")