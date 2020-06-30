from __future__ import division
from collections import defaultdict

# Book: Data Science From Scratch by Joel Grus
# page 28 (pdf file)

# Title: Salaries And Experience 
# Teorical Objective: Learn how to orginize data in buckets from a tuple list an get
# average of the values by bucket.
# Practical Objective: Get the average salary for each tenure bucket of Data Scientist.

# What have I learned from this py file:
#   - From a list of tuples (salary, tenure) got the average salary by tenure
#   - Bucket the tenures (make classes upon  value of tenure)
#   - Organize the data in this buckets
#   - Get the average salary by bucket

# List of tuples (salary, tenure):
salaries_and_tenures = [(83000, 8.7), (88000, 8.1),
                        (48000, 0.7), (76000, 6),
                        (69000, 6.5), (76000, 7.5),
                        (60000, 2.5), (83000, 10),
                        (48000, 1.9), (63000, 4.2)]

# keys are years, values are lists of the salaries for each tenure
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# print("salary_by_tenure:")
# print(salary_by_tenure)

# keys are years, value is average salary for that tenure
average_salary_by_tenure = {
    tenure : sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

# print("\naverage_salary_by_tenure:")
# print(average_salary_by_tenure)

# Bucket the teanures
def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"

# group together the salaries to the correspondant bucket:
# keys are tenure buckets, values are list of salaries for that bucket
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# print("\nsalary_by_tenure_bucket:")
# print(salary_by_tenure_bucket)

# compute the average salary for each group
# keys are tenure buckets, values are average salary for that bucket
average_salary_by_bucket = {
    tenure_bucket : sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}

# print("\naverage_salary_by_bucket:")
# print(average_salary_by_bucket)