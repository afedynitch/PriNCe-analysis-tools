import numpy as np


def convert_to_namedtuple(dictionary, name="GenericNamedTuple"):
    """Converts a dictionary to a named tuple."""
    from collections import namedtuple

    return namedtuple(name, list(dictionary.keys()))(**dictionary)


class XmaxSimple(object):
    """Class implementing the Xmax approximation as in arXiv:1301.6637"""

    EPOS = {
        "X0": 809.7,  # +- 0.3  # g cm**-2
        "D": 62.2,  # +- 0.5  # g cm**-2
        "xi": 0.78,  # +- 0.24 # g cm**-2
        "delta": 0.08,  # +- 0.21 # g cm**-2
        "p0": 3279,  # +- 51   # g**2 cm**-4
        "p1": -47,  # +- 66   # g**2 cm**-4
        "p2": 228,  # +- 108  # g**2 cm**-4
        "a0": -0.461,  # +- 0.006
        "a1": -0.0041,  # +- 0.0016
        "b": 0.059,  # +- 0.002
        "E0": 1e10,  # GeV
    }
    EPOS = convert_to_namedtuple(EPOS, name="EPOS")

    Sybill = {
        "X0": 795.1,  # +- 0.3  # g cm**-2
        "D": 57.7,  # +- 0.5  # g cm**-2
        "xi": -0.04,  # +- 0.24 # g cm**-2
        "delta": -0.04,  # +- 0.21 # g cm**-2
        "p0": 2785,  # +- 46   # g**2 cm**-4
        "p1": -364,  # +- 58   # g**2 cm**-4
        "p2": 152,  # +- 93   # g**2 cm**-4
        "a0": -0.368,  # +- 0.008
        "a1": -0.0049,  # +- 0.0023
        "b": 0.039,  # +- 0.002
        "E0": 1e10,  # GeV
    }
    Sybill = convert_to_namedtuple(Sybill, name="Sybill")

    QGSJet01 = {
        "X0": 774.2,  # +- 0.3  # g cm**-2
        "D": 49.7,  # +- 0.5  # g cm**-2
        "xi": -0.30,  # +- 0.24 # g cm**-2
        "delta": 1.92,  # +- 0.21 # g cm**-2
        "p0": 3852,  # +- 55   # g**2 cm**-4
        "p1": -274,  # +- 70   # g**2 cm**-4
        "p2": 169,  # +- 116  # g**2 cm**-4
        "a0": -0.451,  # +- 0.006
        "a1": -0.0020,  # +- 0.0016
        "b": 0.057,  # +- 0.001
        "E0": 1e10,  # GeV
    }
    QGSJet01 = convert_to_namedtuple(QGSJet01, name="QGSJet01")

    QGSJetII = {
        "X0": 781.8,  # +- 0.3  # g cm**-2
        "D": 45.8,  # +- 0.5  # g cm**-2
        "xi": -1.13,  # +- 0.24 # g cm**-2
        "delta": 1.71,  # +- 0.21 # g cm**-2
        "p0": 3163,  # +- 49   # g**2 cm**-4
        "p1": -237,  # +- 61   # g**2 cm**-4
        "p2": 60,  # +- 100  # g**2 cm**-4
        "a0": -0.386,  # +- 0.007
        "a1": -0.0006,  # +- 0.0021
        "b": 0.043,  # +- 0.002
        "E0": 1e10,  # GeV
    }
    QGSJetII = convert_to_namedtuple(QGSJetII, name="QGSJetII")

    # -------------------------------------------------------------
    # Note JH: The folling 3 parameter sets are Auger internal
    #          Check for permission before using in publications
    # -------------------------------------------------------------

    Sibyll23 = {
        "X0": 824.3,  # +- 0.2  # g cm**-2
        "D": 58.4,  # +- 0.2  # g cm**-2
        "xi": -0.38,  # +- 0.08 # g cm**-2
        "delta": 0.59,  # +- 0.06 # g cm**-2
        "p0": 3810,  # +- 27  # g**2 cm**-4
        "p1": -405,  # +- 30   # g**2 cm**-4
        "p2": 125,  # +- 23  # g**2 cm**-4
        "a0": -0.406,  # +- 0.003
        "a1": -0.0016,  # +- 0.0006
        "b": 0.047,  # +- 0.001
        "E0": 1e10,  # GeV
    }
    Sibyll23 = convert_to_namedtuple(Sibyll23, name="Sibyll23")

    EPOSLHC = {
        "X0": 806.0,  # +- 0.3  # g cm**-2
        "D": 56.3,  # +- 0.2  # g cm**-2
        "xi": 0.35,  # +- 0.12 # g cm**-2
        "delta": 1.04,  # +- 0.10 # g cm**-2
        "p0": 3269,  # +- 39  # g**2 cm**-4
        "p1": -305,  # +- 42   # g**2 cm**-4
        "p2": 124,  # +- 32  # g**2 cm**-4
        "a0": -0.460,  # +- 0.005
        "a1": -0.0006,  # +- 0.0007
        "b": 0.058,  # +- 0.001
        "E0": 1e10,  # GeV
    }
    EPOSLHC = convert_to_namedtuple(EPOSLHC, name="EPOSLHC")

    QGSJetII04 = {
        "X0": 790.1,  # +- 0.3  # g cm**-2
        "D": 54.2,  # +- 0.2  # g cm**-2
        "xi": -0.42,  # +- 0.12 # g cm**-2
        "delta": 0.69,  # +- 0.10 # g cm**-2
        "p0": 3688,  # +- 41  # g**2 cm**-4
        "p1": -428,  # +- 42   # g**2 cm**-4
        "p2": 51,  # +- 32  # g**2 cm**-4
        "a0": -0.394,  # +- 0.005
        "a1": 0.0013,  # +- 0.0009
        "b": 0.045,  # +- 0.001
        "E0": 1e10,  # GeV
    }
    QGSJetII04 = convert_to_namedtuple(QGSJetII04, name="QGSJetII04")

    def __init__(self, model=None):
        if model is None:
            self.model = self.EPOSLHC
        else:
            self.model = model

    def get_mean_Xmax(self, mean_lnA, E):
        """Returns the mean X_max at energy E for mass number A"""
        m = self.model

        # following ref eq. (2.1)
        Xmax_proton = m.X0 + m.D * np.log10(E / m.E0)
        # following ref eq. (2.5)
        fE = m.xi - m.D / np.log(10) + m.delta * np.log10(E / m.E0)

        # following ref eq. (2.6)
        return Xmax_proton + fE * mean_lnA

    def get_sigma2_Xmax_old(self, mean_lnA, sigma2_lnA, E):
        """Returns the squared standard deviation of X_max at energy E for mass number A"""
        mean_lnA2 = sigma2_lnA + mean_lnA**2
        m = self.model

        # following ref eq. (2.5)
        fE = m.xi - m.D / np.log(10) + m.delta * np.log10(E / m.E0)

        # following ref eq. (2.9)
        sigma2_proton = (
            m.p0 + m.p1 * np.log10(E / m.E0) + m.p2 * np.log10(E / m.E0) ** 2
        )
        a = m.a0 + m.a1 * np.log10(E / m.E0)

        # following ref eq. (2.11)
        mean_sigma2_sh = sigma2_proton * (1 + a * mean_lnA + m.b * mean_lnA2)

        # following ref eq. (2.7)
        return mean_sigma2_sh + fE**2 * sigma2_lnA, mean_sigma2_sh

    def get_var_Xmax(self, mean_lnA, var_lnA, E):
        """Returns the squared standard deviation of X_max at energy E for mass number A"""
        m = self.model

        # following ref eq. (2.5)
        fE = m.xi - m.D / np.log(10) + m.delta * np.log10(E / m.E0)

        # following ref eq. (2.9)
        var_proton = m.p0 + m.p1 * np.log10(E / m.E0) + m.p2 * np.log10(E / m.E0) ** 2
        a = m.a0 + m.a1 * np.log10(E / m.E0)

        # following ref eq. (2.11)
        mean_lnA2 = var_lnA + mean_lnA**2
        mean_var_sh = var_proton * (1 + a * mean_lnA + m.b * mean_lnA2)

        # following ref eq. (2.7)
        return mean_var_sh + fE**2 * var_lnA, mean_var_sh


class XmaxGumble(object):
    """Class implementing the Xmax approximation as in arXiv:1305.2331"""

    # tuples are for each model (a0, a1, a2, b0, b1, b2, c0, c1, c2)
    QGSJetII = {
        "mu": (758.444, -10.692, -1.253, 48.892, 0.02, 0.179, -2.346, 0.348, -0.086),
        "sigma": (39.033, 7.452, -2.176, 4.390, -1.688, 0.170),
        "lambda": (0.857, 0.686, -0.040, 0.179, 0.076, -0.0130),
    }
    QGSJetII04 = {
        "mu": (761.383, -11.719, -1.372, 57.344, -1.731, 0.309, -0.355, 0.273, -0.137),
        "sigma": (35.221, 12.335, -2.889, 0.307, -1.147, 0.271),
        "lambda": (0.673, 0.694, -0.007, 0.060, -0.019, 0.017),
    }
    Sibyll21 = {
        "mu": (770.104, -15.873, -0.960, 58.668, -0.124, -0.023, -1.423, 0.977, -0.191),
        "sigma": (31.717, 1.335, -0.601, -1.912, 0.007, 0.086),
        "lambda": (0.683, 0.278, 0.012, 0.008, 0.051, 0.003),
    }
    Epos199 = {
        "mu": (780.013, -11.488, -1.906, 61.911, -0.098, 0.038, -0.405, 0.163, -0.095),
        "sigma": (28.853, 8.104, -1.924, -0.083, -0.961, 0.215),
        "lambda": (0.538, 0.524, 0.047, 0.009, 0.023, 0.010),
    }
    EposLHC = {
        "mu": (775.589, -7.047, -2.427, 57.589, -0.743, 0.214, -0.820, -0.169, -0.027),
        "sigma": (29.403, 13.553, -3.154, 0.096, -0.961, 0.150),
        "lambda": (0.563, 0.711, 0.058, 0.039, 0.067, -0.004),
    }
    E_ref = 1e10  # GeV, the same for all the models

    def __init__(self, model=None):
        if model is None:
            self.model = self.EposLHC
        else:
            self.model = model

        self.p = self.get_poly_params()

    def get_poly_params(self):
        """get the p1,p2,p3 params as in eq. (3.4 - 3.6)"""
        pmodel = {}
        for key in self.model:
            coeff = self.model[key]
            # print key
            if key == "mu":
                a = coeff[0:3]
                b = coeff[3:6]
                c = coeff[6:9]
                p0 = np.poly1d(a[::-1])
                p1 = np.poly1d(b[::-1])
                p2 = np.poly1d(c[::-1])
                # print a, b, c
                # print p0
                # print p1
                # print p2
                pmodel[key] = (p0, p1, p2)
            else:
                a = coeff[0:3]
                b = coeff[3:6]
                p0 = np.poly1d(a[::-1])
                p1 = np.poly1d(b[::-1])
                # print a, b
                # print p0
                # print p1
                pmodel[key] = (p0, p1)
        return pmodel

    def get_gumble_params(self, A, E):
        egrid = np.log10(E / self.E_ref)
        lnA = np.log(A)

        p = self.p["mu"]
        mu = p[0](lnA) + p[1](lnA) * egrid + p[2](lnA) * egrid**2

        p = self.p["sigma"]
        sigma = p[0](lnA) + p[1](lnA) * egrid

        p = self.p["lambda"]
        lamb = p[0](lnA) + p[1](lnA) * egrid

        return mu, sigma, lamb
