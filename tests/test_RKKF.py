# -*- coding: utf-8 -*-
"""
Tests are run using pytest.
Tests for Reactor_Kinetics_Kalman_filter.py
"""

def test_import():
    exc = None
    try:
        from Reactor_Kinetics_Kalman_Filter import main
    except Exception as e:
        exc = e
    assert exc is None

def test_noexceptions():
    from Reactor_Kinetics_Kalman_Filter import main
    time_spacings=[10]
    exceptions, figures = main(time_spacings=time_spacings)
    assert exceptions == []

def test_figures():
    from Reactor_Kinetics_Kalman_Filter import main
    time_spacings = [10]
    exceptions, figures = main(time_spacings=time_spacings)
    assert len(figures) == len(time_spacings)
