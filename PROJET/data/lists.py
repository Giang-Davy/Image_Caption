#!/usr/bin/env python3
import ast
import os

# Food list
food_path = os.path.join(os.path.dirname(__file__), "food_list.txt")
with open(food_path, "r", encoding="utf-8") as f:
    contenu = f.read()
list_food = ast.literal_eval(contenu)

# Animal list
animal_list = ["dog", "dogs", "cat", "cats", "horse", "horses", "cow", "cows",
"sheep", "goat", "goats", "pig", "pigs", "chicken", "chickens", "duck", "ducks",
"bird", "birds", "rabbit", "rabbits", "mouse", "mice", "rat", "rats", "elephant", 
"elephants", "lion", "lions", "tiger", "tigers", "bear", "bears", "zebra", "zebras",
"giraffe", "giraffes", "monkey", "monkeys", "ape", "apes", "gorilla", "gorillas",
"kangaroo", "kangaroos", "wolf", "wolves", "fox", "foxes", "deer", "camel", "camels",
"donkey", "donkeys", "buffalo", "buffaloes", "leopard", "leopards", "cheetah",
"cheetahs", "crocodile", "crocodiles", "alligator", "alligators", "snake", "snakes",
"lizard", "lizards", "turtle", "turtles", "frog", "frogs", "fish", "whale", "whales",
"dolphin", "dolphins", "shark", "sharks", "seal", "seals", "penguin", "penguins",
"owl", "owls", "eagle", "eagles", "hawk", "hawks", "parrot", "parrots", "goose",
"geese", "turkey", "turkeys", "swan", "swans"]
