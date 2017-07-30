#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `dwglasso` package."""


import unittest

import numpy as np
# from matplotlib import pyplot as plt
# import seaborn

from .context import dwglasso
from dwglasso import var


class TestVar_basic(unittest.TestCase):
    """Tests for `dwglasso` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.B0 = np.array([[0.8, 0.1],
                            [0., 0.8]])
        self.B1 = np.array([[0.1, 0.0],
                            [0., 0.1]])
        self.B = [self.B0, self.B1]
        self.G = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        return

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_is_stable(self):
        stable_system = var.VAR(self.B)
        self.assertTrue(stable_system.is_stable())

        unstable_system = var.VAR([2 * self.B0, self.B1])
        self.assertFalse(unstable_system.is_stable())
        return

    def test_001_induced_graph(self):
        system = var.VAR(self.B)
        self.assertTrue(np.all(system.induced_graph() == self.G))
        return

    def test_002_drive1(self):
        system = var.VAR(self.B)
        u = np.array([1., 1.])
        y = system.drive(u)
        self.assertTrue(np.all(y == u))
        return

    def test_003_drive2(self):
        system = var.VAR(self.B)
        U = np.ones((3, 2))
        Y_expected = np.array([[1.,    1.],
                               [1.8,  1.9],
                               [2.54, 2.8]])
        Y = system.drive(U)

        self.assertTrue(np.allclose(Y, Y_expected, atol=1e-12),
                        msg='Y = %s,\n Y_expeted = %s' % (Y, Y_expected))
        return

    def test_004_drive3(self):
        system = var.VAR(self.B)
        u = np.array([1., 1.])
        U = np.ones((3, 2))
        Y_expected = system.drive(U)
        system.reset()
        for t in range(3):
            y = system.drive(u)
            self.assertTrue(np.allclose(Y_expected[t, :],
                                        y, atol=1e-12),
                            msg='y = %s,\n y_expected = %s' %
                            (y, Y_expected[t, :]))
        return

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'dwglasso.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output
