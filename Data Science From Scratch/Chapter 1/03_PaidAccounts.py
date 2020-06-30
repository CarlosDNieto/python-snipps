# Book: Data Science From Scratch by Joel Grus
# page 31 (pdf file)

# Title: Paid Accounts
# Teorical Objective: Not much
# Practical Objective: Know which user pay for accounts and which don't 

# What have I learned from this py file:
#   - 

# List of paid accounts, tuples (years_of_experience, paid_or_unpaid)
paid_accounts = [(0.7, "paid"),
                (1.9, "unpaid"),
                (2.5, "paid"),
                (4.2, "unpaid"),
                (6, "unpaid"),
                (6.5, "unpaid"),
                (7.5, "unpaid"),
                (8.1, "unpaid"),
                (8.7, "paid"),
                (10, "paid")]

def predict_paid_or_unpaid(years_of_experience):
    if years_of_experience < 3.0:
        return "paid"
    elif years_of_experience < 8.5:
        return "unpaid"
    else:
        return "paid"

