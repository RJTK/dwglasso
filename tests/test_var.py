#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `dwglasso` package."""


import unittest

import numpy as np
from matplotlib import pyplot as plt
import seaborn

from .context import dwglasso
from dwglasso import var
from .random_var_generators import random_var, iid_gaussian_var, iid_ber_graph


class TestVAR_plots(unittest.TestCase):
    '''Creates some plots from a VAR system'''
    @classmethod
    def setUpClass(cls):
        cls.B0 = np.array([[0.8, 0.1],
                           [0., 0.8]])
        cls.B1 = np.array([[0.1, 0.0],
                           [0., 0.1]])
        cls.B = [cls.B0, cls.B1]
        np.random.seed(2718)
        return

    def test_000_plot(self):
        '''Drive a stable system with 0 mean noise'''
        T = 100
        U = np.random.multivariate_normal(np.zeros(2),
                                          0.1 * np.eye(2),
                                          T)
        t = range(T)
        system = var.VAR(self.B)
        Y = system.drive(U)
        for i in range(2):
            plt.plot(t, U[:, i], linestyle='--', alpha=0.5,
                     label='$u_%d(t)$' % i)
            plt.plot(t, Y[:, i], linewidth=2, label='$x_%d(t)$' % i)
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('Output')
        plt.title('Driven Stable VAR(2) System')
        plt.show()
        return

    def test_001_plot(self):
        '''Drive an unstable system with 0 mean noise'''
        T = 10
        U = np.random.multivariate_normal(np.zeros(2),
                                          0.1 * np.eye(2),
                                          T)
        t = range(T)
        system = var.VAR([2 * B_tau for B_tau in self.B])
        Y = system.drive(U)
        for i in range(2):
            plt.plot(t, U[:, i], linestyle='--', alpha=0.5,
                     label='$u_%d(t)$' % i)
            plt.plot(t, Y[:, i], linewidth=2, label='$x_%d(t)$' % i)
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('Output')
        plt.title('Driven Unstable VAR(2) System')
        plt.show()
        return


class TestVAR(unittest.TestCase):
    """Basic tests for VAR model"""
    # setUpClass / tearDownClass are executed only once for TestVAR
    @classmethod
    def setUpClass(cls):
        # Data common to many tests
        cls.B0 = np.array([[0.8, 0.1],
                           [0., 0.8]])
        cls.B1 = np.array([[0.1, 0.0],
                           [0., 0.1]])
        cls.B = [cls.B0, cls.B1]
        cls.G = np.array([[1.0, 1.0],
                          [0.0, 1.0]])

        np.random.seed(2718)
        cls.n = 50
        cls.p = 4
        cls.q = 0.4
        cls.B_random = random_var(lambda: iid_ber_graph(cls.n, cls.q),
                                  lambda G: iid_gaussian_var(cls.p, cls.G,
                                                             0.65 / cls.q),
                                  max_tries=1)
        return

    # setUp / tearDown are executed before and after every test
    def setUp(self):
        """Set up test fixtures, if any."""
        return

    def tearDown(self):
        """Tear down test fixtures, if any."""
        return

    def test_000_basic_init(self):
        system = var.VAR(self.B)
        self.assertEqual(system.n, 2)
        self.assertEqual(system.p, 2)
        self.assertEqual(system.t, 0)
        return

    def test_001_is_stable(self):
        stable_system = var.VAR(self.B)
        self.assertTrue(stable_system.is_stable())

        unstable_system = var.VAR([2 * self.B0, self.B1])
        self.assertFalse(unstable_system.is_stable())
        return

    def test_002_induced_graph(self):
        system = var.VAR(self.B)
        self.assertTrue(np.all(system.induced_graph() == self.G))
        return

    def test_003_drive1(self):
        system = var.VAR(self.B)
        u = np.array([1., 1.])
        y = system.drive(u)
        self.assertTrue(np.all(y == u))
        self.assertEqual(system.t, 1)
        return

    def test_004_drive2(self):
        system = var.VAR(self.B)
        U = np.ones((3, 2))
        Y_expected = np.array([[1.,    1.],
                               [1.8,  1.9],
                               [2.54, 2.8]])
        Y = system.drive(U)

        self.assertTrue(np.allclose(Y, Y_expected, atol=1e-12),
                        msg='Y = %s,\n Y_expeted = %s' % (Y, Y_expected))
        self.assertEqual(system.t, 3)
        return

    def test_005_drive3(self):
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

    def test_006_drive4(self):
        # Same as test_005_drive3, but with a more complicated system
        T = 25
        system = var.VAR(self.B_random)
        n = system.n

        # Random input noise
        U = np.random.multivariate_normal(np.zeros(n),
                                          np.eye(n),
                                          T)
        Y_expected = system.drive(U)
        system.reset()
        for t in range(T):
            y = system.drive(U[t, :])
            self.assertTrue(np.allclose(Y_expected[t, :],
                                        y, atol=1e-12))

        return

    def test_007_state_init(self):
        system = var.VAR(self.B, x_0=np.array([1.8, 1.9, 1., 1.]))
        y = system.drive(np.array([1., 1.]))
        self.assertTrue(np.allclose(y, np.array([2.54, 2.8])))
        return

    def test_008_exceptions(self):
        with self.assertRaises(ValueError):
            var.VAR([np.eye(2), np.eye(3)])
        with self.assertRaises(ValueError):
            var.VAR(self.B, x_0=np.array([1, 2, 3]))
        with self.assertRaises(ValueError):
            system = var.VAR(self.B)
            system.drive(np.array([1, 2, 3]))

    # def test_command_line_interface(self):
    #     """Test the CLI."""
    #     runner = CliRunner()
    #     result = runner.invoke(cli.main)
    #     assert result.exit_code == 0
    #     assert 'dwglasso.cli.main' in result.output
    #     help_result = runner.invoke(cli.main, ['--help'])
    #     assert help_result.exit_code == 0
    #     assert '--help  Show this message and exit.' in help_result.output
