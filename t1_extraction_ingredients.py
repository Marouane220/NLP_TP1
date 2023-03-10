# -*- coding: utf-8 -*-
import re

ingredients_fn = "./data/ingredients.txt"

# Mettre dans cette partie la (les) expression(s) régulière(s)
# que vous utilisez pour analyser les ingrédients
#
# Vos regex ici...
#
pattern = re.compile(
    r"\d*,?\d* (t\S*|c\S*? à (s\S*|c\S*)|ml |g |lb )?(.*\(.*\))?")

def load_ingredients(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw_items = f.readlines()
    ingredients = [x.strip() for x in raw_items]
    return ingredients


def get_ingredients(text):
    # Insérez ici votre code pour l'extraction d'ingrédients.
    # En entrée, on devrait recevoir une ligne de texte qui correspond à un ingrédient.
    # Par ex. 2 cuillères à café de poudre à pâte
    # Vous pouvez ajouter autant de fonctions que vous le souhaitez.
    #
    # IMPORTANT : Ne pas modifier la signature de cette fonction
    #             afin de faciliter notre travail de correction.
    #
    # Votre code ici...
    #
    try:
        quan = re.match(pattern, text).group(0)
        ing = text.replace(quan, "")
        ingred = re.sub(r'\bde\b', '', ing, count=1)
    except:
        return "NAN", "NAN" # si l'expression n'arrive pas à détécter la quantité l'exception est récupéré est des NAN sont retournées
    return quan, ingred   # À modifier - retourner la paire extraite


if __name__ == '__main__':
    # Vous pouvez modifier cette section
    print("Lecture des ingrédients du fichier {}. Voici quelques exemples: ".format(
        ingredients_fn))
    all_items = load_ingredients(ingredients_fn)
    # for item in all_items[:5]:
    #     print("\t", item)
    print("\nExemples d'extraction")
    for item in all_items:
        quantity, ingredient = get_ingredients(item)
        print("\t{}\t QUANTITE:{}   INGREDIENT:{}".format(
            item, quantity, ingredient))
