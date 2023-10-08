import pandas as pd

df = pd.read_csv("compare.csv")

# Go through row by row and ask the user to judge the recipes
# If they are correct, add 1 to the score
# If they are incorrect, do nothing
score = 0

ALL_MEATS = [
    "beef",
    "pork",
    "chicken",
    "breast",
    "turkey",
    "lamb",
    "duck",
    "fillet",
    "bacon",
    "sausage",
    " ham",
    "meat",
    "steak",
    "fish",
    "sirloin",
    "tenderloin",
    "venison",
    "veal",
    "goat",
    "rabbit",
    "quail",
    "pheasant",
    "bison",
    "ribeye",
    "prosciutto",
    "pastrami",
    "salami",
    "pepperoni",
    "jerky",
    "chorizo",
    "tuna",
    "salmon",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "mussel",
    "scallop",
    "squid",
    "octopus",
    "clam",
    "anchovy",
    "sardine",
    "trout",
    "cod",
    "haddock",
    "mackerel",
    "tilapia",
    "sea bass",
    "hake",
    "pollock",
    "catfish",
    "perch",
    "halibut",
    "sole",
    "swordfish",
    "snapper",
    "grouper",
    "carp",
    "fowl",
    "oyster",
    "crabmeat",
    "hot dog"
]

for i in range(len(df)):
    print(i, "-"*45)
    print("INGREDIENTS 1: ")
    print(df["Ingredients 1"][i])
    print("RECIPE 1: ")
    print(df["Recipe 1"][i])

    print("INGREDIENTS 2: ")
    print(df["Ingredients 2"][i])
    print("RECIPE 2: ")
    print(df["Recipe 2"][i])

    print()
    
    has_meat = 0
    for meat in ALL_MEATS:
        if meat in df["Ingredients " + str(df["Healthier"][i])][i]:
            has_meat = 1
            break
    score += has_meat

    # judgement = input("RECIPE 1 OR RECIPE 2: ")
    # if judgement == str(df["Healthier"][i]):
    #     print("MATCH!")
    #     score += 1
    # else:
    #     print("INCORRECT!")
        
    # print("Score: ", score)

    print("-"*50)
    print()

print(score/len(df))
