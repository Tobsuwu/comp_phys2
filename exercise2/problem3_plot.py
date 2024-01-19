""" This program plots a figure of the converged energies calculated in puhti. The energy values are close to the
picture in the handout, excluding the leftmost one. The R values also seem to be half the ones in the handout.
Probably something wrong with the workflow file problem3.py and thus the calculations done in puhti.
The figure itself is quite nice, especially when zoomed in. """

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def main():

    d_eq = 1.2074  # nuclei separation in Angstrom
    conv_E = np.loadtxt('prob3_converged_energies.txt')  # Load energies calculated in puhti from text file
    diffs = np.linspace(d_eq - 0.15, d_eq + 0.15, len(conv_E))  # Creating the deviations close to equilibrium value.

    # Spline interpolation using scipy
    f = interpolate.InterpolatedUnivariateSpline(diffs, conv_E)
    diffs_new = np.linspace(d_eq - 0.15, d_eq + 0.15, 100)
    spline = f(diffs_new)

    # Polyfit using numpy polyfit
    z = np.polyfit(diffs, conv_E, 3)
    p = np.poly1d(z)

    # Plotting the figure
    plt.figure()
    plt.plot(diffs, conv_E, 'o', diffs_new, p(diffs_new), '-', diffs_new, spline, '--')
    plt.xlabel('R (units of Ångström)')
    plt.ylabel('E (units of Rydberg)')
    plt.title('Potential Energy Surface of 02')
    plt.legend(['Points form puhti', 'Polyfit', 'Spline interpolation'])
    plt.show()


main()