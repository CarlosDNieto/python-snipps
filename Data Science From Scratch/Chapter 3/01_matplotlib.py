from __future__ import division
import matplotlib.pyplot as plt
from collections import Counter

# * Book: Data Science From Scratch by Joel Grus
# * page 73 (pdf file)

# * Title: Python Crash Curse
# Teorical Objective: Learn the basics of matplotlib
# Practical Objective: Plot nice graphs

# * What have I learned from this py file:
#   -

# *  A simple plot
years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# create a line chart, years on x-axis, gdp on y-axis
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# add a title
plt.title("Nominal GDP")

# add a label to the y-axis
plt.ylabel("Billions of $")
#plt.show()


# * Bar charts
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# bars are by default width 0.8, so we'll add 0.1 to the left coordinates
# so that each bar is centered
xs = [i + 0.1 for i, _ in enumerate(movies)]

# plot bars with left x-coordinates [xs], heights [num_oscars]
plt.bar(xs, num_oscars)
plt.ylabel("# of Academy Awards")
plt.title("My Favorite Movies")

# label x-axis with movie names at bar centers
plt.xticks([i + 0.1 for i, _ in enumerate(movies)], movies)
#plt.show()

# this are also good for plotting histograms of bucketed numeric values
grades = [83,95,91,87,70,0,85,82,100,67,73,77,0]
decile = lambda grade: (grade // 10) * 10
histogram = Counter(decile(grade) for grade in grades)

plt.bar([x - 0.1 for x in histogram.keys()],                      # shift each bar to the left by 4
        histogram.values(),                                     # give each bar its correct height
        8)                                                      # give each bar a width of 8

plt.axis([-5, 105, 0, 5])                                       # x-axis from -5 to 105,
                                                                # y-axis from 0 to 5

plt.xticks([10 * i for i in range(11)], histogram.keys())                         # x-axis labels at 0, 10, ..., 100
plt.xlabel("Decile")
plt.ylabel("# of Students")
plt.title("Distribution of Exam 1 Grades")
# plt.show()

# another example
mentions = [500, 505]
years = [2013, 2014]

plt.bar([2013, 2014], mentions)
plt.xticks(ticks=years, labels=years)
plt.ylabel("# of times I heard someone say 'data science'")

# if you don't do this, matplotlib will label the x-axis 0, 1
# and then add a +2.013e3 off in the corner (bad matplotlib!)
#plt.ticklabel_format(useOffset=False)

# misleading y-axis only shows the part above 500
plt.axis([2012.5,2014.5,499,506])
plt.title("Look at the 'Huge' Increase!")
# plt.show()


# * Line Charts
# Example:
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]

# we can make multiple calls to plt.plot
# to show multiple series on the same chart
plt.plot(xs, variance, 'g-', label='variance')                  # green solid line
plt.plot(xs, bias_squared, 'r-.', label='bias^2')               # red dot-dashed line
plt.plot(xs, total_error, 'b:', label='total error')            # blue dotted line

# because we've assigned labels to each series
# we can get a legend for free
# loc=9 means "top center"
plt.legend(loc=9)
plt.xlabel("model complexity")
plt.title("The Bias-Variance Tradeoff")
# plt.show()
