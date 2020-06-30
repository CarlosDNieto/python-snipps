from __future__ import division                     # integer division is lame
from collections import Counter
from collections import defaultdict

# * Book: Data Science From Scratch by Joel Grus

# * This intro in from the page 22 (pdf page)
# * Title: Data Scientist You May Know Suggester
# Teorical Objective: Finding Key Connectors
# Practical Objective: Make a "people you make know" suggester based on the mutual friends
# and mutual interests.

# * What have I learned from this py file:
#   - Make a "Data Scientist You May Know" suggester for a Data Science Social Network.
#   - Make a list of "friends" for each user from a conection list of tuples.
#   - Managed to show friends of a friend list from friendship list of tuples.
#   - Transform a list of tuples (user_id, interest) to a dictionary {interest:user_ids}
#   - Transform a list of tuples (user_id, interest) to a dictionary {user_id:interests}

# List of users with id and name
users = [
    { "id": 0, "name": "Hero" },
    { "id": 1, "name": "Dunn" },
    { "id": 2, "name": "Sue" },
    { "id": 3, "name": "Chi" },
    { "id": 4, "name": "Thor" },
    { "id": 5, "name": "Clive" },
    { "id": 6, "name": "Hicks" },
    { "id": 7, "name": "Devin" },
    { "id": 8, "name": "Kate" },
    { "id": 9, "name": "Klein" }
]

# List of friendships, paired id's
friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# Add a list of friends for each user
for user in users:
    user["friends"] = []

# Populate the friends lists with friendships data
for i, j in friendships:
    users[i]["friends"].append(users[j])            # add i as a friend of j
    users[j]["friends"].append(users[i])            # add j as a friend of i

def number_of_friends(user):
    """How many friends does the _user_ have?"""
    return len(user["friends"])                     # length of the friends_id list

total_conections = sum(number_of_friends(user) for user in users) # 24

num_users = len(users)                              # length of the users list
avg_connections = total_conections / num_users      # 2.4

# create a list (user_id, number_of_friends)
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]

# each pair is (user_id, num_friends)
# [(1, 3), (2, 3), (3, 3), (5, 3), (8, 3),
# (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]

def friends_of_friend_ids_bad(user):
    # foaf is short for "friend of a friend"
    return [foaf["id"]
            for friend in user["friends"]           # for each of user friends
            for foaf in friend["friends"]]          # for each of _their_ friends

# When we call this on users[0] (Hero), it produces:
# [0, 2, 3, 0, 1, 3]
print("friends_of_friend_ids_bad(users[0]) output:")
print(friends_of_friend_ids_bad(users[0]))

# # it includes 2 times the user_id 0 and 2 times the user_id 3 
# print [friend["id"] for friend in users[0]["friends"]] # [1, 2]
# print [friend["id"] for friend in users[1]["friends"]] # [0, 2, 3]
# print [friend["id"] for friend in users[2]["friends"]] # [0, 1, 3]

def not_the_same(user, other_user):
    """Two users are not the same if they have different ids"""
    return user["id"] != other_user["id"]

def not_friends(user, other_user):
    """other_user is not a friend if he's not in user["friends"];
    that is, if he's not_the_same as all the people in user["friends"]"""
    return all(not_the_same(friend, other_user)
                for friend in user["friends"])

def friends_of_friend_ids(user):
    return Counter(foaf["id"]
                    for friend in user["friends"]
                    for foaf in friend["friends"]
                    if not_the_same(user, foaf)
                    and not_friends(user, foaf))

print("\nfriends_of_friend_ids(users[0]) output:")
print(friends_of_friend_ids(users[0]))

# list of interests (user_id, interest):
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"),
    (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"),
    (1, "Postgres"), (2, "Python"), (2, "scikit-learn"), (2, "scipy"),
    (2, "numpy"), (2, "statsmodels"), (2, "pandas"), (3, "R"), (3, "Python"),
    (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"),
    (4, "libsvm"), (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"),
    (5, "Haskell"), (5, "programming languages"), (6, "statistics"),
    (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"),
    (7, "neural networks"), (8, "neural networks"), (8, "deep learning"),
    (8, "Big Data"), (8, "artificial intelligence"), (9, "Hadoop"),
    (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# Find users with a certain interest
def data_scientist_who_like(target_interest):
    return [user_id 
            for user_id, user_interest in interests
            if user_interest == target_interest]

# this works but it has to examine the whole list of interests for every search.
# So, we'll better off building an index from interests to users

# dictionary of user_ids by interest
# keys are interests, values are lists of user_ids with that interest
user_ids_by_interest = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)

# dictionare of interests by user_id
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

# Now is easy to find interests in common between users
# - Iterate over the users interests.
# - For each interest iterate over the over users with that interest.
# - Keep count of how many times we see each other user.

def most_common_interests_with(user):
    return Counter(interested_user_id
                    for interest in interests_by_user_id[user["id"]]
                    for interested_user_id in user_ids_by_interest[interest]
                    if interested_user_id != user["id"])

print(most_common_interests_with(users[0]))