import numpy as np
import pickle

users_file = 'users.dat'
movies_file = 'movies.dat'
ratings_file = 'ratings.dat'

user_ids = []
with open(users_file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if parts:
            user_ids.append(int(parts[0]))
user_ids = sorted(user_ids)

movie_ids = []
with open(movies_file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if parts:
            movie_ids.append(int(parts[0]))
movie_ids = sorted(movie_ids)

movieID_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}
num_movies = len(movie_ids)

user_dict = {}
for uid in user_ids:
    user_dict[uid] = np.full(num_movies, np.nan, dtype=float)

with open(ratings_file, 'r', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split('::')
        if len(parts) >= 3:
            uid = int(parts[0])
            mid = int(parts[1])
            rating = float(parts[2])
            idx = movieID_to_index[mid]
            user_dict[uid][idx] = rating

# 1000 testing, rest training
test_users = user_ids[:1000]
train_users = user_ids[1000:]

with open('user_dict.pkl', 'wb') as f:
    pickle.dump(user_dict, f)

with open('train_users.pkl', 'wb') as f:
    pickle.dump(train_users, f)

with open('test_users.pkl', 'wb') as f:
    pickle.dump(test_users, f)

print("Pickling complete.")
print("- user_dict.pkl")
print("- train_users.pkl")
print("- test_users.pkl")
