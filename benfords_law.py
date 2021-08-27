from typing import Union, Tuple, Dict
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stats_modified import chisquare

temp_actuals = {
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 0,
    '6': 0,
    '7': 0,
    '8': 0,
    '9': 0,
}


class BenfordsLaw:
    """
    Newcomb-Benford's Law Analysis
    Takes a list/array of numbers representing some real world dataset of numbers
    and analyzes to asses whether it fits the Newcomb-Benford's Law (also known as
    the Law of Analogous Numbers). Fit is currently determined by either running a statistical goodness-of-fit test,
    or by running a visual test by plotting the actual distribution of first-significant digits in the dataset
    against the expected distribution according to Benford's Law.
    """

    def __init__(self, data: Union[list, np.array, pd.Series],col=None,control=None,control_val=None):
        """
        Initialize Benford's Law Analysis object
        :param data: Dataset of numbers to test Newcomb-Benford's Law against.
        """
        self.expected_distribution = {
            '1': 0.301,
            '2': 0.176,
            '3': 0.125,
            '4': 0.097,
            '5': 0.079,
            '6': 0.067,
            '7': 0.058,
            '8': 0.051,
            '9': 0.046,
        }
        self.col = col
        self.control = control
        self.control_val = control_val
        self.fsd_counts, self.fsd_distribution = dict(), dict()
        try:
            # enforce that all numbers passed can be evaluated into a numeric representation
            # handles the non-essential case where numbers are negative
            self.data = np.array([abs(float(x)) for x in data])

            # remove unnecessary null values if any
            self.data = self.data[np.logical_not(np.isnan(self.data))]
        except ValueError:
            print('All values must be numerical')
            raise

    def _get_fsd(self, number: float):
        # this handles for integers and decimals like in one run
        # the possible cost is increasing the run time for integers without the need to
        # however, the increase in run time for ints should be negligible especially to the end user
        if number > 0:
            fsd = str(float(str(number).replace('.', '')))[0]
            return fsd

    def _extract_fsd(self):
        self.fsd = [self._get_fsd(number) for number in self.data if self._get_fsd(number) is not None]

    def get_counts(self) -> Dict[str, int]:
        """
        Get frequency of first significant digits passed in the dataset.
        :return: key pair value of each first significant digit and it's respective frequency
        """
        temp_counts = deepcopy(temp_actuals)
        self.fsd_counts = dict((x, self.fsd.count(x)) for x in set(self.fsd))
        temp_counts.update(self.fsd_counts)
        self.fsd_counts = temp_counts
        return self.fsd_counts

    def get_distribution(self) -> Dict[str, float]:
        """
        Get percentage distribution of first significant digits passed in the dataset
        :return: key pair value of each first significant digit and it's
                 respective percentage
        """
        temp_distribution = deepcopy(temp_actuals)
        self.fsd_distribution = dict((x, (self.fsd.count(x) / len(self.fsd))) for x in set(self.fsd))
        temp_distribution.update(self.fsd_distribution)
        self.fsd_distribution = temp_distribution
        return self.fsd_distribution

    def prepare_actual_distribution(self,
                                    get_fsd_counts: bool = False):
        self._extract_fsd()
        self.get_distribution()
        if get_fsd_counts:
            self.get_counts()

    def apply_visual_test(self,stat,
                          figsize: Tuple[int, int] = (15, 7)):
        """
        Plot first significant digit distribution against the expectation of Benford's Law
        :param figsize: Dimensions of the figure to plot in the format: (width, height)
        """
        barWidth = 0.25

        r1 = np.arange(len(self.expected_distribution.values()))
        r2 = [x + barWidth for x in r1]

        # Make the plot
        plt.figure(figsize=figsize)
        plt.bar(r1, self.expected_distribution.values(), color='blue', width=barWidth, edgecolor='white',
                label='expected')
        plt.bar(r2, self.fsd_distribution.values(), color='red', width=barWidth, edgecolor='white', label='actual')

        # Add xticks on the middle of the group bars
        plt.xlabel('First Significant Digit', fontweight='bold')
        plt.ylabel('Distribution', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(self.expected_distribution.values()))],
                   self.expected_distribution.keys())
        plt.ylim(0,1)
        plt.text(7.5,0.8,'# data points:' + str(len(self.fsd)))
        plt.text(7.2,0.7,'(chi2,p): (' + str(round(stat[0], 2)) + ',' + str(round(stat[1],4)) + ')')
        # Create legend & Show graphic
        plt.legend()
        try:
            plt.title("First Significant Digit distribution vs Expected Benford's Law Distribution for " + self.col + " with " + self.control + "=" + str(self.control_val))
            plt.savefig('results/' + self.col + '_' + self.control + '=' + str(self.control_val) + '.png')
        except:
            plt.title("First Significant Digit distribution vs Expected Benford's Law Distribution for chlor_a")
            plt.savefig('results/chlor_a.png')
        plt.show()

    def apply_chi_sq_test(self,
                          alpha=0.05) -> Tuple[float, float]:
        """
        Apply Chi-Squared Goodness of fit test to test if the dataset's first significant digit
        distribution meets the expectation of Benford's Law. It passes the test if the
        p-value is greater than specified alpha and fails otherwise.
        :param alpha: Optional. Specifies the required significance level based on which the null hypothesis is rejected or failed to reject. Default = 0.05
        :return: Chi-Squared statistic, p-value
        """
        expected_counts = {k: int(v * len(self.fsd)) for k, v in self.expected_distribution.items()}
        print(list(expected_counts.values()),sum(list(expected_counts.values())))
        print(list(self.fsd_counts.values()),sum(list(self.fsd_counts.values())))
        #print(sum(list(self.fsd_counts.values())), sum(list(expected_counts.values())))
        statistic, p_value = chisquare(f_obs=list(self.fsd_counts.values()), f_exp=list(expected_counts.values()))
        if p_value > alpha:
            test_status = 'passed'
        else:
            test_status = 'failed'
        print(f'Chi-squared test {test_status} with statistic: {statistic} and p-value: {p_value}')
        return (statistic, p_value)

    def apply_benfords_law(self):
        """
        Runs all relevant processes and then applies all tests to input dataset
        """
        self._extract_fsd()
        print('fsd extracted')
        self.get_counts()
        print('counts got')
        self.get_distribution()
        print('distribution got')
        stat = self.apply_chi_sq_test()
        self.apply_visual_test(stat)
